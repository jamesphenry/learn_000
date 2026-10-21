#
# # Test Runner in a C header
#
# ## test-runner.sh -- Shell Script - Runner
#
# @description Test Runner in a C header
# @author [Jon Ruttan](jonruttan@gmail.com)
# @copyright 2020 Jon Ruttan
# @license MIT No Attribution (MIT-0)
#
# ## Usage
#
# ### Run All Tests
#
# ```bash
# sh test-runner.sh
# ```
#
SCRIPT="$0"
CC=${CC:-cc}
INCLUDE="$(dirname "$SCRIPT")/include"
CFLAGS=${CFLAGS:-"-Wall -Wextra -Wno-unused-parameter"}
CFLAGS="${CFLAGS} -g -fdiagnostics-color=always -DDEBUG -DTESTS -I$INCLUDE"

# The tests are run under valgrind when it is installed, and valgrind is the
# one that says what is wrong with the memory. The error exit code is what
# turns a memory error into a failed test, so it is asked for.
if [ -z "$VALGRIND" ]; then
	if [ -x "$(which valgrind)" ]; then
		VALGRIND="valgrind --quiet --show-error-list=yes --leak-check=full --show-leak-kinds=all --error-exitcode=1"
	fi
fi
RUNNER=${RUNNER:-${VALGRIND}}

TESTS="$@"

# If $TESTS is a directory search it for tests
if [ -d "$TESTS" ]; then
	TESTS="$TESTS/*-test.c"
fi

# Determine correct `stat` formatting parameters
# Use GNU stat formatting
STAT="stat -c '%Y %n'"
# Try to stat this file, on error use *BSD formatting
eval $STAT "$0" 1>/dev/null 2>/dev/null || STAT="stat -f '%m %N'"

# Sort by Modified date
TESTS=$(eval $STAT $TESTS | sort -t ' ' -nk1 | cut -d ' ' -f2-)

# The test programs are built in a directory of their own, so that two runs
# of the tests at the same time do not build over one another.
BUILD_DIR=".test-binaries-$$"
mkdir -p "$BUILD_DIR"

failed=0

for test in $TESTS
do
	echo "${test}"
	out="$BUILD_DIR/$(basename ${test}.out)"
	if $CC $CFLAGS $SOURCES "${test}" -o "${out}"; then
		if ! $RUNNER "${out}"; then
			echo "FAILED: ${test}"
			failed=1
		fi
	else
		echo "FAILED TO BUILD: ${test}"
		failed=1
	fi
	[ -x ] && rm "${out}"
done

rm -rf "$BUILD_DIR"

exit $failed
