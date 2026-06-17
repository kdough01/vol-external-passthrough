import sys
import asyncio
import dagger

async def main():

    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        src = client.host().directory(
            "..",
            exclude=[
                "build/",
                "*.h5",
                ".git/",
                "__pycache__/",
                "ci/*.pyc",
            ]
        )

        base = (
            client.host()
            .directory(".", exclude=["build/", "*.h5", ".git/", "__pycache__/", "ci/*.pyc"])
            .docker_build(dockerfile="ci/Dockerfile")
        )

        print("\n ----- Build -----")

        build = (
            base
            .with_directory("/workspace", src)
            .with_workdir("/workspace")
            .with_exec([
                "cmake", "-S", ".", "-B", "build",
                "-DCMAKE_C_COMPILER=mpicc",
                "-DUSE_CUDA=OFF",
                "-DCMAKE_PREFIX_PATH=/usr/local",
                "-DCMAKE_BUILD_TYPE=Release",
            ])
            .with_exec([
                "cmake", "--build", "build", "-j4"
            ])
        )

        try:
            await build.stdout()
            print("Build PASSED")
        except dagger.ExecError as e:
                print(f"Build: FAILED\n{e.stderr}")
                sys.exit(1)

        print("\n ----- CI Tests -----")

        test = (
            build
            .with_workdir("/workspace/build")
            .with_env_variable("HDF5_VOL_CONNECTOR", "pass_through_ext under_vol=0;under_info={}")
            .with_env_variable("HDF5_PLUGIN_PATH", "/workspace/build")
            .with_env_variable("HDF5_VOL_PRESSIO_COMPRESSOR", "noop")
            .with_env_variable("HDF5_VOL_PRESSIO_LEVEL", "1")
            .with_exec(["./tests/test_ci"])
        )

        try:
            output = await test.stdout()
            print(output)

            if "FAIL:" in output:
                print("CI Tests: FAILED")
                sys.exit(1)

            else:
                print("CI Tests PASSED")

        except dagger.ExecError as e:
            print(f"CI Tests FAILED\n{e.stdout}\n{e.stderr}")
            sys.exit(1)

        print("\n ----- SZ3 Default Compressor -----")

        test = (
            build
            .with_workdir("/workspace/build")
            .with_env_variable("HDF5_VOL_CONNECTOR", "pass_through_ext under_vol=0;under_info={}")
            .with_env_variable("HDF5_PLUGIN_PATH", "/workspace/build")
            .with_env_variable("HDF5_VOL_PRESSIO_COMPRESSOR", "sz3")
            .with_env_variable("HDF5_VOL_PRESSIO_LEVEL", "1")
            .with_exec(["./tests/test_ci"])
        )

        try:
            output = await test.stdout()
            print(output)

            if "FAIL:" in output:
                print("CI Tests: FAILED")
                sys.exit(1)

            else:
                print("CI Tests PASSED")

        except dagger.ExecError as e:
            print(f"CI Tests FAILED\n{e.stdout}\n{e.stderr}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())