# HDF5 Compression VOL: A Compressor Agnostic HDF5 VOL Adaptor for GPU-Accelerated Compression

This is the public repo for Compression VOL. The primary goal of this VOL Adaptor is to enable new classes of compression techniques to HDF5, including but not limited to progressive decompression and GPU compression.

The design, implementation, and performance evaluation of Compression VOL will be documented in a paper soon.

A more detailed further discussion of design decisions and code walk-through can be found in the [Wiki](https://github.com/kdough01/vol-external-passthrough/wiki) of this Github repo. There you will find an extensive breakdown of function calls that were added or modified, expanded details on how to turn on compression metrics, a description of our error handling calls and wrappers, as well as some basic examples of how to use this adaptor.

The rest of this README will provide an overview of the features of this VOL, files in this repository, and a quick setup guide for how to get build the adaptor and get started.

## Features

- CPU and GPU-accelerated comrpession through Libpressio backend
- Benchmarking tools for compression throughput and overhead analysis
- Configurable chunking strategies

## Files

Files can be split into three categories: base, compression, error handling. Base files can be thought of as the starter code for a VOL Adaptor; this is where we intercept the data in the create, open, read, and write functions. While these files have been modified, they maintained their overall structure. Compression files contain the function calls necessary to perform the compression.

## Setup Guide

### Dependencies

### Build

### Usage

## Citation