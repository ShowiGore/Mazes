# ==============================================================================
# Mazes Project - Root Makefile
# ==============================================================================

CMAKE ?= /usr/bin/cmake
BUILD_DIR ?= build
BIN ?= $(BUILD_DIR)/Mazes

# Execution Parameters
H ?= 8193
W ?= 8193
GEN ?=
SOLVERS ?=
SEED ?=
ALPHA ?=
PNG ?= 0
BIN_SAVE ?= 0
LOAD ?=

.PHONY: all build clean run generate benchmark solve help

all: build

# ------------------------------------------------------------------------------
# Build Targets
# ------------------------------------------------------------------------------

build:
	@mkdir -p $(BUILD_DIR)
	@$(CMAKE) -B $(BUILD_DIR) -S .
	@$(CMAKE) --build $(BUILD_DIR) -j

clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

# ------------------------------------------------------------------------------
# Execution Targets
# ------------------------------------------------------------------------------

# General run target
run: build
	@CMD="$(BIN)"; \
	if [ -n "$(LOAD)" ]; then \
		CMD="$$CMD --load $(LOAD)"; \
	else \
		if [ -z "$(GEN)" ]; then \
			echo "Error: Generator not specified. Provide GEN=<name> (e.g. make run GEN=houston)"; \
			exit 1; \
		fi; \
		CMD="$$CMD -g $(GEN) -H $(H) -W $(W)"; \
		if [ -n "$(SEED)" ]; then CMD="$$CMD -s $(SEED)"; fi; \
		if [ -n "$(ALPHA)" ]; then CMD="$$CMD -a $(ALPHA)"; fi; \
	fi; \
	if [ -n "$(SOLVERS)" ]; then CMD="$$CMD --solvers $(SOLVERS)"; fi; \
	if [ "$(PNG)" = "1" ]; then CMD="$$CMD --png"; fi; \
	if [ "$(BIN_SAVE)" = "1" ]; then CMD="$$CMD --bin"; fi; \
	$$CMD

# Generate a maze without solving (solving is optional)
generate: build
	@if [ -z "$(GEN)" ]; then \
		echo "Error: Generator not specified. Provide GEN=<name> (e.g. make generate GEN=houston)"; \
		exit 1; \
	fi; \
	CMD="$(BIN) -g $(GEN) -H $(H) -W $(W)"; \
	if [ -n "$(SEED)" ]; then CMD="$$CMD -s $(SEED)"; fi; \
	if [ -n "$(ALPHA)" ]; then CMD="$$CMD -a $(ALPHA)"; fi; \
	if [ "$(PNG)" = "1" ]; then CMD="$$CMD --png"; fi; \
	if [ "$(BIN_SAVE)" = "1" ]; then CMD="$$CMD --bin"; fi; \
	$$CMD

# Benchmark all solvers on a generated maze
benchmark: build
	@if [ -z "$(GEN)" ]; then \
		echo "Error: Generator not specified. Provide GEN=<name> (e.g. make benchmark GEN=houston)"; \
		exit 1; \
	fi; \
	CMD="$(BIN) -g $(GEN) -H $(H) -W $(W) --solvers all"; \
	if [ -n "$(SEED)" ]; then CMD="$$CMD -s $(SEED)"; fi; \
	if [ -n "$(ALPHA)" ]; then CMD="$$CMD -a $(ALPHA)"; fi; \
	if [ "$(PNG)" = "1" ]; then CMD="$$CMD --png"; fi; \
	if [ "$(BIN_SAVE)" = "1" ]; then CMD="$$CMD --bin"; fi; \
	$$CMD

# Solve an existing binary maze
solve: build
	@if [ -z "$(LOAD)" ]; then \
		echo "Error: LOAD=<path_to_maze> required (e.g. make solve LOAD=generated_mazes/binary/...maze SOLVERS=bidir-gbfs)"; \
		exit 1; \
	fi; \
	CMD="$(BIN) --load $(LOAD)"; \
	if [ -n "$(SOLVERS)" ]; then CMD="$$CMD --solvers $(SOLVERS)"; else CMD="$$CMD --solvers bidir-gbfs"; fi; \
	if [ "$(PNG)" = "1" ]; then CMD="$$CMD --png"; fi; \
	if [ "$(BIN_SAVE)" = "1" ]; then CMD="$$CMD --bin"; fi; \
	$$CMD

# ------------------------------------------------------------------------------
# Help & Usage Guide
# ------------------------------------------------------------------------------

help:
	@echo "=============================================================================="
	@echo " Mazes Engine - Makefile Commands & Options"
	@echo "=============================================================================="
	@echo " Build Targets:"
	@echo "   make build               Compile project using CMake in build/"
	@echo "   make clean               Remove build artifacts"
	@echo ""
	@echo " Execution Targets:"
	@echo "   make run GEN=<name>      Run maze engine with options"
	@echo "   make generate GEN=<name> Generate a maze without solving"
	@echo "   make benchmark GEN=<name>Generate a maze and run all solvers for comparison"
	@echo "   make solve LOAD=<file>   Solve an existing binary .maze file"
	@echo ""
	@echo " Configurable Variables:"
	@echo "   GEN=<name>               Generator (REQUIRED for generation):"
	@echo "                              houston, wilson, aldous-broder, recursive-division, fractal"
	@echo "   SOLVERS=<list>           Solvers (comma-separated, 'all', or 'none'):"
	@echo "                              bidir-gbfs, gbfs, dead-end, gpu-dead-end, gpu-bidir-bfs, astar, recursive, all"
	@echo "   H=<int>, W=<int>         Height and Width (default: 8193)"
	@echo "   SEED=<uint>              Random seed (default: random)"
	@echo "   ALPHA=<float>            Transition threshold for Houston (default: 0.333)"
	@echo "   PNG=1                    Save PNG image(s) to generated_mazes/images/"
	@echo "   BIN_SAVE=1               Save compact binary .maze / .sol file(s) to generated_mazes/binary/"
	@echo "   LOAD=<path>              Path to .maze file for decoupled solving"
	@echo ""
	@echo " Examples:"
	@echo "   make generate GEN=houston H=2001 W=2001 PNG=1"
	@echo "   make run GEN=houston H=2001 W=2001 SOLVERS=bidir-gbfs PNG=1"
	@echo "   make benchmark GEN=wilson H=1001 W=1001"
	@echo "   make solve LOAD=generated_mazes/binary/42_2001_2001_houston.maze SOLVERS=bidir-gbfs"
	@echo "=============================================================================="
