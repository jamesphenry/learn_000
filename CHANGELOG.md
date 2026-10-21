# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

  - Added the game of Nim, in `games/nim/`, with its own tests.
  - Added a league of a hundred models in the game of Nim, which is written
    up in `LEAGUE.md`.
  - Added the whole table of the league to the demonstration, which comes out
    of it when the word `table` is given as well.
  - Added a perfect player that needs no memory of its own: the perfect
    player for Nim works the answer out from the exclusive or of the heaps.
  - Added `make test-valgrind`, which runs the tests under valgrind and says
    so when valgrind is not installed.
  - Added directory structure.
  - Added `.editorconfig` (https://editorconfig.org/).
  - Added `CHANGELOG.md` (https://keepachangelog.com/en/1.1.0/).
  - Added test-runner submodule.
  - Added source files.
  - Added `Makefile`.
  - Added `.gitignore`.
  - Added `LICENSE.md`.
  - Added `README.md`.

### Changed

  - The games were put behind an interface: the framework in `include/` and
    `src/` knows nothing about any particular game, and each game has its own
    directory under `games/`.
  - The tests are run under valgrind when it is installed, and a memory error
    fails the build.
  - The demonstration plays the game of Nim as well as tic-tac-toe.
