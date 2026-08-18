/* sperr_trunc_probe.cpp
 *
 * Answers one question mechanically: for a given volume + SPERR chunk
 * configuration, is a SPERR 3D bitstream safe to truncate by taking a
 * contiguous prefix of its bytes?
 *
 * This is the invariant the VOL read plan currently assumes. SPERR only
 * guarantees it for single-chunk streams; for multi-chunk streams
 * progressive_truncate() must locate each chunk's sub-bitstream, so a
 * prefix is not enough.
 *
 * Three variants are compared at each percentage:
 *
 *   A  reference : sperr_trunc_3d(full_buf,  full_len,  pct)
 *   B  short     : sperr_trunc_3d(prefix,    prefix_len, pct)
 *                  -- a naive prefix reader. Expected to FAIL when multi-chunk.
 *   C  zero-tail : sperr_trunc_3d(padded,    full_len,  pct)
 *                  -- prefix bytes, rest memset to 0, full length declared.
 *                  THIS IS WHAT vol_container_plan + vol_container_fetch DO.
 *                  Dangerous case: it can return success and decode to garbage.
 *
 * Each truncated stream is decompressed and scored against the original.
 * If C's PSNR tracks A's, prefix reading is safe for this configuration.
 * If C succeeds but its PSNR collapses, the VOL is silently serving garbage.
 *
 * Build:
 *   g++ -O2 -std=c++17 sperr_trunc_probe.cpp -o sperr_trunc_probe \
 *       -I<sperr-prefix>/include -L<sperr-prefix>/lib -lSPERR -Wl,-rpath,<sperr-prefix>/lib
 *
 * Run (synthetic volume, SPERR's default chunking):
 *   ./sperr_trunc_probe --dims 256 256 256
 *
 * Force a single chunk (the configuration where prefix truncation is legal):
 *   ./sperr_trunc_probe --dims 256 256 256 --chunk 256 256 256
 *
 * Multi-chunk, i.e. what you probably have today:
 *   ./sperr_trunc_probe --dims 512 512 512 --chunk 128 128 128
 *
 * Real data:
 *   ./sperr_trunc_probe --dims X Y Z --input vol.f32 [--double]
 */

#include <SPERR_C_API.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned kPcts[] = {5, 10, 25, 50, 75, 100};

struct Config {
    size_t dimx = 256, dimy = 256, dimz = 256;
    /* 0 means "let SPERR pick", which is its 256^3 default. */
    size_t chunkx = 0, chunky = 0, chunkz = 0;
    int    is_float = 1;
    int    mode = 3;          /* 3 = fixed point-wise error */
    double quality = 1e-6;
    size_t nthreads = 1;
    /* Mirrors VOL_SPERR_SLACK in vol_progressive.h. */
    size_t slack = 4096;
    std::string input;
};

/* Smooth field plus a sharper feature, so truncation artifacts are visible
 * rather than hidden in noise that SPERR would discard anyway. */
std::vector<float> synth(size_t nx, size_t ny, size_t nz)
{
    std::vector<float> v(nx * ny * nz);
    for (size_t k = 0; k < nz; k++)
        for (size_t j = 0; j < ny; j++)
            for (size_t i = 0; i < nx; i++) {
                const double x = double(i) / nx, y = double(j) / ny, z = double(k) / nz;
                double s = std::sin(6.28318530718 * 3 * x) *
                           std::cos(6.28318530718 * 2 * y) *
                           std::sin(6.28318530718 * 4 * z);
                double r = std::sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5));
                double blob = std::exp(-40.0 * r * r);
                v[(k * ny + j) * nx + i] = float(s + 3.0 * blob);
            }
    return v;
}

struct Score { double psnr, rmse, maxerr; };

Score score(const float *ref, const float *got, size_t n)
{
    double se = 0.0, mx = 0.0, lo = ref[0], hi = ref[0];
    for (size_t i = 0; i < n; i++) {
        const double d = double(ref[i]) - double(got[i]);
        se += d * d;
        if (std::fabs(d) > mx) mx = std::fabs(d);
        if (ref[i] < lo) lo = ref[i];
        if (ref[i] > hi) hi = ref[i];
    }
    const double rmse  = std::sqrt(se / double(n));
    const double range = (hi > lo) ? (hi - lo) : 1.0;
    const double psnr  = (rmse > 0.0) ? 20.0 * std::log10(range / rmse)
                                      : std::numeric_limits<double>::infinity();
    return { psnr, rmse, mx };
}

/* Truncate, decode, score. Returns false if either step failed. */
bool try_variant(const void *src, size_t src_len, unsigned pct,
                 const float *ref, size_t nelem, size_t nthreads,
                 Score *out, size_t *trunc_bytes)
{
    void  *tbuf = nullptr;   /* MUST be null: sperr_trunc_3d returns 1 otherwise */
    size_t tlen = 0;

    if (C_API::sperr_trunc_3d(src, src_len, pct, &tbuf, &tlen) != 0 || !tbuf || !tlen) {
        std::free(tbuf);
        return false;
    }
    *trunc_bytes = tlen;

    void  *dbuf = nullptr;
    size_t dx = 0, dy = 0, dz = 0;
    const int rc = C_API::sperr_decomp_3d(tbuf, tlen, 1 /*float*/, nthreads,
                                          &dx, &dy, &dz, &dbuf);
    std::free(tbuf);
    if (rc != 0 || !dbuf) { std::free(dbuf); return false; }

    if (dx * dy * dz != nelem) { std::free(dbuf); return false; }
    *out = score(ref, static_cast<const float *>(dbuf), nelem);
    std::free(dbuf);
    return true;
}

void usage(const char *p)
{
    std::fprintf(stderr,
        "usage: %s [--dims X Y Z] [--chunk CX CY CZ] [--input FILE] [--double]\n"
        "          [--mode M] [--quality Q] [--threads N] [--slack B]\n"
        "  --chunk omitted lets SPERR use its default (256^3).\n"
        "  --mode 1=fixed BPP, 2=fixed PSNR, 3=fixed point-wise error (default)\n", p);
}

} /* namespace */

int main(int argc, char **argv)
{
    Config c;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--dims" && i + 3 < argc) {
            c.dimx = std::strtoul(argv[++i], nullptr, 10);
            c.dimy = std::strtoul(argv[++i], nullptr, 10);
            c.dimz = std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--chunk" && i + 3 < argc) {
            c.chunkx = std::strtoul(argv[++i], nullptr, 10);
            c.chunky = std::strtoul(argv[++i], nullptr, 10);
            c.chunkz = std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--input" && i + 1 < argc) {
            c.input = argv[++i];
        } else if (a == "--double") {
            c.is_float = 0;
        } else if (a == "--mode" && i + 1 < argc) {
            c.mode = std::atoi(argv[++i]);
        } else if (a == "--quality" && i + 1 < argc) {
            c.quality = std::atof(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            c.nthreads = std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--slack" && i + 1 < argc) {
            c.slack = std::strtoul(argv[++i], nullptr, 10);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (c.chunkx == 0) { c.chunkx = 256; c.chunky = 256; c.chunkz = 256; }

    const size_t nelem = c.dimx * c.dimy * c.dimz;

    /* ---- source volume ---- */
    std::vector<float> vol;
    if (!c.input.empty()) {
        std::FILE *f = std::fopen(c.input.c_str(), "rb");
        if (!f) { std::perror("open input"); return 1; }
        if (c.is_float) {
            vol.resize(nelem);
            if (std::fread(vol.data(), sizeof(float), nelem, f) != nelem) {
                std::fprintf(stderr, "short read: expected %zu floats\n", nelem);
                std::fclose(f); return 1;
            }
        } else {
            std::vector<double> d(nelem);
            if (std::fread(d.data(), sizeof(double), nelem, f) != nelem) {
                std::fprintf(stderr, "short read: expected %zu doubles\n", nelem);
                std::fclose(f); return 1;
            }
            vol.assign(d.begin(), d.end());   /* score in float */
        }
        std::fclose(f);
    } else {
        vol = synth(c.dimx, c.dimy, c.dimz);
    }

    /* ---- compress once, as libpressio/the VOL would ---- */
    void  *stream = nullptr;
    size_t stream_len = 0;
    const int crc = C_API::sperr_comp_3d(vol.data(), 1 /*is_float, we hold floats*/,
                                         c.dimx, c.dimy, c.dimz,
                                         c.chunkx, c.chunky, c.chunkz,
                                         c.mode, c.quality, c.nthreads,
                                         &stream, &stream_len);
    if (crc != 0 || !stream) {
        std::fprintf(stderr, "sperr_comp_3d failed rc=%d\n", crc);
        return 1;
    }

    const size_t nchunks_x = (c.dimx + c.chunkx - 1) / c.chunkx;
    const size_t nchunks_y = (c.dimy + c.chunky - 1) / c.chunky;
    const size_t nchunks_z = (c.dimz + c.chunkz - 1) / c.chunkz;
    const size_t nchunks   = nchunks_x * nchunks_y * nchunks_z;

    std::printf("volume      : %zu x %zu x %zu  (%zu elems, %.1f MiB raw f32)\n",
                c.dimx, c.dimy, c.dimz, nelem, nelem * 4.0 / 1048576.0);
    std::printf("chunk dims  : %zu x %zu x %zu\n", c.chunkx, c.chunky, c.chunkz);
    std::printf("chunk grid  : %zu x %zu x %zu = %zu chunk(s)%s\n",
                nchunks_x, nchunks_y, nchunks_z, nchunks,
                nchunks == 1 ? "   <- prefix truncation is legal here"
                             : "   <- prefix truncation is NOT legal here");
    std::printf("stream      : %zu bytes (%.2fx)\n",
                stream_len, (nelem * 4.0) / double(stream_len));
    std::printf("slack       : %zu bytes\n\n", c.slack);

    std::printf("%4s | %-28s | %-28s | %-28s\n",
                "pct", "A reference (full stream)",
                "B short prefix (naive)", "C zero-tail (YOUR VOL)");
    std::printf("%4s-+-%-28s-+-%-28s-+-%-28s\n",
                "----", std::string(28, '-').c_str(),
                std::string(28, '-').c_str(), std::string(28, '-').c_str());

    int verdict_ok = 1;

    for (unsigned pct : kPcts) {
        /* The byte range vol_container_plan would fetch. */
        size_t need = (stream_len * size_t(pct)) / 100 + c.slack;
        if (need > stream_len) need = stream_len;

        Score  sa{}, sb{}, sc{};
        size_t ta = 0, tb = 0, tc = 0;

        const bool oka = try_variant(stream, stream_len, pct,
                                     vol.data(), nelem, c.nthreads, &sa, &ta);

        const bool okb = try_variant(stream, need, pct,
                                     vol.data(), nelem, c.nthreads, &sb, &tb);

        std::vector<unsigned char> padded(stream_len, 0);
        std::memcpy(padded.data(), stream, need);
        const bool okc = try_variant(padded.data(), stream_len, pct,
                                     vol.data(), nelem, c.nthreads, &sc, &tc);

        char ba[32], bb[32], bc[32];
        if (oka) std::snprintf(ba, sizeof ba, "%7.2f dB  %8zu B", sa.psnr, ta);
        else     std::snprintf(ba, sizeof ba, "FAILED");
        if (okb) std::snprintf(bb, sizeof bb, "%7.2f dB  %8zu B", sb.psnr, tb);
        else     std::snprintf(bb, sizeof bb, "FAILED (expected if >1 chunk)");
        if (okc) std::snprintf(bc, sizeof bc, "%7.2f dB  %8zu B", sc.psnr, tc);
        else     std::snprintf(bc, sizeof bc, "FAILED");

        std::printf("%4u | %-28s | %-28s | %-28s\n", pct, ba, bb, bc);

        /* C is what the VOL does. It must both succeed and match A. */
        if (pct < 100) {
            if (!okc) verdict_ok = 0;
            else if (oka && (sa.psnr - sc.psnr) > 0.5) verdict_ok = 0;
        }
    }

    std::printf("\n");
    if (nchunks == 1 && verdict_ok) {
        std::printf("VERDICT: single chunk, and the zero-tail read matches the reference.\n"
                    "         The current VOL prefix plan is correct FOR THIS CONFIG ONLY.\n");
    } else if (!verdict_ok) {
        std::printf("VERDICT: BROKEN. The zero-tail read (what vol_container_fetch produces)\n"
                    "         either fails or decodes to materially worse data than the\n"
                    "         reference. Per-chunk byte ranges are required.\n");
    } else {
        std::printf("VERDICT: multi-chunk but zero-tail happened to match. Do not rely on\n"
                    "         this -- it is not a guarantee SPERR makes. Re-run with other\n"
                    "         dims/quality before trusting it.\n");
    }

    std::free(stream);
    return verdict_ok ? 0 : 1;
}
