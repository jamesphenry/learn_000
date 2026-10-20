# c-skeleton

A minimalist project skeleton for C development. This repository provides a basic structure to set up C projects quickly and efficiently.

## Introduction

c-skeleton is a simple and minimalistic starting point for C projects. It includes the essential files and directory structure needed to begin developing C applications.

## Features

- Basic project structure
- Flexible Makefile for building the project
- Sample source files
- Directory setup for source, header, and test files

## Prerequisites

Ensure you have the following installed on your system:

- **Standard C Compiler**
- **Make**
- **AWK** (optional, for `make help`)
- **fswatch** (optional, for `make watch`)

## Installation

Clone the repository to your local machine:

```sh
git clone --recurse-submodules https://github.com/jonruttan/c-skeleton.git
cd c-skeleton
```

## Usage

To build the project, simply run:

```sh
make
```

For help on make options, run:

```sh
make help
```


To run all tests:

```sh
make test
```

To run the tests with AddressSanitizer and UndefinedBehaviorSanitizer:

```sh
make test-asan
```

To run a specific test:

```sh
TST=test/main.test.c make test
```

To watch files for changes:

```sh
make watch
```

To clean up the compiled files, use:

```sh
make clean
```

## Project Structure

```
learn_000
├── include
│   ├── board.h
│   ├── layer.h
│   ├── main.h
│   ├── neuron.h
│   └── ...
├── src
│   ├── board.c
│   ├── layer.c
│   ├── main.c
│   ├── neuron.c
│   └── ...
├── test
│   ├── board.test.c
│   ├── layer.test.c
│   ├── main.test.c
│   ├── network.test.c
│   ├── neuron.test.c
│   ├── thread.test.c
│   └── ...
├── obj
│   └── ...
├── .editorconfig
├── .gitignore
├── .gitmodules
├── CHANGELOG.md
├── LICENSE.md
├── Makefile
└── README.md
```

- **include/**: Contains the header files (.h files).
- **src/**: Contains the source files (.c files). Each file is compiled and linked on its own, so new modules can be added without changing the Makefile.
- **test/**: Contains the test files (.test.c files).
- **obj/**: Contains the compiled object files, created by the build.
- **.editorconfig**: A file to define editor coding styles (https://editorconfig.org/).
- **.git**: Version control directory.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **.gitmodules**: Defines submodule properties.
- **CHANGELOG.md**: Changelog (https://keepachangelog.com/en/1.1.0/).
- **LICENSE**: License.
- **Makefile**: A file to automate the build process.
- **README.md**: This file.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
