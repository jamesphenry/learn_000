.POSIX:

# Directories
INC_DIR = include
SRC_DIR = src
OBJ_DIR = obj
TST_DIR = test
TMP_DIR = tmp
BIN_DIR = src
GAMES_DIR = games

# Compiler and flags
CC ?= gcc
CFLAGS += -I$(INC_DIR) -Wall -Wextra -Werror

# Libraries to link against. The math functions in <math.h> live in libm. The
# test runner compiles and links in a single step and puts the compiler flags
# before the source files, and the linker only keeps a shared library that
# resolves a symbol it has already seen, so the library is marked as needed
# to survive that order.
LDLIBS += -Wl,--no-as-needed -lm
CFLAGS += $(LDLIBS)

# Every game has its own include directory and its own sources, and every
# header in them is on the path, so the framework and the games can be built
# and tested on their own.
GAME_INC = $(wildcard $(GAMES_DIR)/*/include)
GAME_SRC = $(wildcard $(GAMES_DIR)/*/src/*.c)
GAME_TST = $(wildcard $(GAMES_DIR)/*/test/*.test.c)

CFLAGS += $(foreach dir,$(GAME_INC),-I$(dir))

# Every .c file under src/ and under the games is a separate translation unit
# and is compiled and linked on its own, so the project is free to grow more
# sources (board.c, main.c, ...) without touching this file.
INC = $(wildcard $(INC_DIR)/*.h)
SRC = $(wildcard $(SRC_DIR)/*.c) $(GAME_SRC)
TST = $(wildcard $(TST_DIR)/*.test.c) $(GAME_TST)
OBJ = $(SRC:%.c=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/learn_000
DEBUG_TARGET = $(TARGET)-debug


# Default target
all: $(TARGET) ## Build default target
.PHONY: all


# Create obj, bin and tmp directories if they don't exist
$(OBJ_DIR) $(BIN_DIR) $(TMP_DIR):
	@mkdir -p $@

# Rule to build the target executable
$(TARGET): $(OBJ) | $(BIN_DIR)
	$(CC) $(OBJ) -o $@ $(CFLAGS) $(LDFLAGS)

# Rule to build object files
$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


# Build target
target: $(TARGET) ## Build target
.PHONY: target

# Build debug target
debug: $(DEBUG_TARGET) ## Build debug target
.PHONY: debug

# Rules to build debug target
$(DEBUG_TARGET): CFLAGS += -g -Og -DDEBUG
$(DEBUG_TARGET): LDFLAGS += -g
$(DEBUG_TARGET): $(OBJ) | $(BIN_DIR)
	$(CC) $(OBJ) -o $@ $(CFLAGS) $(LDFLAGS)


# Run target
run: $(TARGET) ## Build and run target
	$(TARGET)

# Run debug target
run-debug: $(DEBUG_TARGET) ## Build and run debug target
	$(DEBUG_TARGET)


# The tests are run under valgrind when it is installed. A memory error is a
# failed test: valgrind is asked to fail the run when it finds one, and the
# output is checked as well, so that nothing slips through.
VALGRIND_FLAGS = --quiet --show-error-list=yes --leak-check=full --show-leak-kinds=all --error-exitcode=1
TEST_ERRORS = \*\*FAIL\*\*|Segmentation fault|SIGSEGV|core dumped|Invalid read|Invalid write|Invalid free|Mismatched free|uninitialised|definitely lost|indirectly lost|possibly lost|ERROR SUMMARY: [1-9]

# Run tests
#
# The test files are compiled on their own, without $(OBJ), and pull the
# sources in with #include instead (see test/main.test.c). That keeps every
# source in the same translation unit as the test, which is what lets the
# test override puts() to capture the greeting. A new source file therefore
# has to be added to the test file as well.
#
# The runner says which test failed, and the output is checked as well
# before the build is called a success.
test: ## Run tests
	@if out="$$(CFLAGS="$(CFLAGS) -g -Og -I. -DTEST -pthread" sh test/test-runner/test-runner.sh $(TST) 2>&1)"; then rc=0; else rc=$$?; fi; \
	echo "$$out"; \
	if [ $$rc -ne 0 ] || echo "$$out" | grep -qE '$(TEST_ERRORS)'; then \
		echo "tests failed"; \
		exit 1; \
	fi
.PHONY: test

# Run tests under valgrind
#
# The tests are run under valgrind when it is installed, but this target says
# that valgrind is wanted, and says so when it is not there.
test-valgrind: ## Run tests under valgrind
	@command -v valgrind >/dev/null 2>&1 || { echo "valgrind is not installed"; exit 1; }; \
	$(MAKE) test VALGRIND="valgrind $(VALGRIND_FLAGS)"
.PHONY: test-valgrind

# Run tests with AddressSanitizer and UndefinedBehaviorSanitizer
test-asan: ## Run tests with AddressSanitizer and UndefinedBehaviorSanitizer
	@out="$$(RUNNER=command CFLAGS="$(CFLAGS) -g -Og -I. -DTEST -pthread -fsanitize=address,undefined" sh test/test-runner/test-runner.sh $(TST) 2>&1)"; rc=$$?; \
	echo "$$out"; \
	if [ $$rc -ne 0 ] || echo "$$out" | grep -qE '$(TEST_ERRORS)'; then \
		echo "tests failed"; \
		exit 1; \
	fi
.PHONY: test-asan

# Run tests with fast runner (no valgrind)
test-quick: ## Run fast tests (no Valgrind)
	@out="$$(RUNNER=command CFLAGS="$(CFLAGS) -g -Og -I. -DTEST -pthread" sh test/test-runner/test-runner.sh $(TST) 2>&1)"; rc=$$?; \
	echo "$$out"; \
	if [ $$rc -ne 0 ] || echo "$$out" | grep -qE '$(TEST_ERRORS)'; then \
		echo "tests failed"; \
		exit 1; \
	fi
.PHONY: test-quick

# Alias test
tests: test ## Run tests (alias)
.PHONY: tests

# Alias test-quick
tests-quick: test-quick ## Run fast tests (no Valgrind) (alias)
.PHONY: tests-quick


# Lint sources
lint: ## Lint sources
	$(CC) -fsyntax-only $(CFLAGS) -g -Wall -pedantic $(SRC)
.PHONY: lint


# Run valgrind on target
valgrind: ## Run Valgrind on target
	$(CC) $(CFLAGS) -g -Wall $(SRC) && valgrind -v --leak-check=full ./a.out && rm a.out
.PHONY: valgrind


# Strip symbols from target
strip: $(TARGET) ## Strip symbols from target
	strip $(TARGET)
.PHONY: strip


# Clean target
clean: ## Clean compiled files
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET) $(DEBUG_TARGET)
.PHONY: clean


# Watch source for changes
watch: ## Watch source for changes
	while true; do \
		fswatch -o --event Created --event Updated --event MovedTo Makefile $(INC) $(SRC) $(TST) | \
		while read num; do \
			make && make test; \
		done; \
	done
.PHONY: watch

# Display help
help: ## Display this help section
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[32m%-38s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
.PHONY: help
