SHELL := /bin/bash
YEAR = $$(date +%Y)
LAST_YEAR = $$(($(YEAR) - 1))
COLOR_LOG = \033[34m
COLOR_INFO = \033[32m
COLOR_WARNING = \033[33m
COLOR_ERROR = \033[31m
COLOR_END = \033[0m

SRC_DIRS = corenet tests

.PHONY: format install-githooks prepush-check update-copyright-year

update-copyright-year:
	@if [[ "$$CHECK_ONLY" == "" ]]; then \
		if ! command -v rg &>/dev/null; then \
			printf "$(COLOR_INFO)Installing ripgrep ...$(COLOR_END)\n"; \
			if [[ "$$OSTYPE" == "darwin"* ]]; then \
				brew install ripgrep; \
			else \
				sudo --preserve-env apt-get install ripgrep; \
			fi; \
		fi; \
		rg -l "\(C\) $(LAST_YEAR)" \
			| grep -v "ACKNOWLEDGEMENTS" \
			| xargs -r -n1 -P 10 sed -i.bak.copyright "s/(C) $(LAST_YEAR) Apple/(C) $(YEAR) Apple/g" \
			&& find . -name '*.bak.copyright' \
			| xargs -r -n1 -P 10 rm; \
	fi

format: format-init-files format-license-headers format-eof-new-lines format-isort format-black format-conventions
	# Formatting checks succeeded.

# The existence of __init__.py files is necessary for the inclusion of packages in the setup.py (i.e. pip-instsallable) package.
format-init-files:
	# Checking missing __init__.py files...
	@python_files=$$(find $(SRC_DIRS) -iname "*.py"); \
	for dir_name in $$(find $(SRC_DIRS) -type d -not -name __pycache__); do \
	    grep -q "^$$dir_name/" <(echo "$$python_files") || continue; \
	    init_file="$$dir_name/__init__.py"; \
	    [[ -f "$$init_file" ]] && continue; \
		if [[ "$$CHECK_ONLY" == "1" ]]; then \
			printf "$(COLOR_ERROR)Missing $$init_file. $(COLOR_END)\n"; \
		else \
			touch "$$init_file"; \
			printf "$(COLOR_LOG)Added $$init_file. $(COLOR_END)\n"; \
		fi \
	done

format-license-headers: update-copyright-year
	# Checking missing license headers files...
# Read the first 4 lines of each file to check if the Copyright notice header exists. 
# The Copyright note is usually found on line 3, but some files may have an additional 
# '#/bin/bash' or '#/usr/bin/env python' header line, hence reading 3+1 lines.
	@for f in $$(find . -iname "*.py"); do \
	    [ -s "$$f" ] || continue; \
		git check-ignore -q "$$f" && continue; \
	    export COPYRIGHT_NOTICE_LINES=4; \
	    (head -$$COPYRIGHT_NOTICE_LINES $$f | grep -q "# *Copyright (C) $(YEAR) Apple Inc. All Rights Reserved.") && continue; \
	    [[ "$$CHECK_ONLY" == "1" ]] && exit 1; \
	    sed -i '' -e "1s/^/#\n# For licensing see accompanying LICENSE file.\n# Copyright (C) $(YEAR) Apple Inc. All Rights Reserved.\n#\n\n/" $$f; \
	    printf "$(COLOR_LOG)Added license header for $$f$(COLOR_END)\n"; \
	done

format-eof-new-lines:
	# Ensure newline at the end of all files...
# Inspired by https://unix.stackexchange.com/a/161853/55814
	@if [[ "$$CHECK_ONLY" == "" ]]; then \
	    git ls-files -z | while IFS= read -rd '' f; do \
	        if [[ -f "$$f" ]] && (file --mime-encoding "$$f" | grep -qv binary); then \
	            tail -c1 < "$$f" | read -r _ || (echo >> "$$f" && printf "$(COLOR_LOG) Added newline at the end of $$f$(COLOR_END)\n"); \
	        fi; \
	    done; \
	fi

format-isort:
	# Running isort...
	@if [[ "$$CHECK_ONLY" == "1" ]]; then \
	    isort --check-only .; \
	else \
	    isort .; \
	fi

format-black:
	# Running black formatter...
	@if [[ "$$CHECK_ONLY" == "1" ]]; then \
	    black --check .; \
	else \
	    black .; \
	fi

format-conventions:
	@if [[ "$$CHECK_ONLY" == "1" ]]; then \
	    echo "# Checking coding conventions..."; \
	    convention_test_files=(tests/test_conventions.py); \
	    [[ -d tests/internal ]] && convention_test_files+=(tests/internal/test_internal_conventions.py); \
		export _parallel_args="$$( (python -c "import xdist.plugin" 2>/dev/null && echo '-n 4') || true)"; \
	    if ! pytest --junit-xml="" -q "$${convention_test_files[@]}" $$_parallel_args; then \
	        printf "$(COLOR_ERROR) Please manually fix the above convention errors. $(COLOR_END)\n"; \
	        exit 1; \
	    fi; \
	fi


prepush-check:
	@printf "$(COLOR_LOG)[pre-push hook]$(COLOR_END)\n"
	@CHECK_ONLY=1 make format || (printf "$(COLOR_ERROR)Formatting checks failed.$(COLOR_END) Please run '$(COLOR_INFO)make format$(COLOR_END)' command, commit, and push again.\n" && exit 1);
	@if [ -n "$$(git status --porcelain)" ]; then \
	    printf "$(COLOR_WARNING)Formatting checks succeeded, but please consider committing UNCOMMITTED changes to the following files:$(COLOR_END)\n"; \
	    git status --short; \
	else \
	    printf "$(COLOR_INFO)Formatting checks succeeded.$(COLOR_END)\n"; \
	fi

install-githooks:
	@echo -e "#!/usr/bin/env bash\n" '\
set -euo pipefail\n\
# Check if Git LFS is installed\n\
if ! command -v git-lfs >/dev/null 2>&1; then\n\
	echo >&2 "Error: Git LFS is not installed. Please install Git LFS to continue."\n\
	exit 1\n\
fi\n\
\n\
git lfs pre-push "$$@"\n\n\
printf "$(COLOR_LOG)[pre-push hook]$(COLOR_END) Running formatting checks and fixes... To skip this hook, please run \"git push --no-verify\".\\n";\n\
if grep -q "^prepush-check:" Makefile 2>/dev/null; then\n\
		make prepush-check;\n\
else\n\
		printf "$(COLOR_WARNING)WARNING:$(COLOR_END) Skipping the pre-push formatting checks and fixes. The git hook is installed (probably on a different git branch), but Makefile is either missing or old on this branch.\\n";\n\
fi \
	' > "$$(git rev-parse --git-path hooks)/pre-push"
	chmod +x "$$(git rev-parse --git-path hooks)/pre-push"
	# Successfully installed the pre-push hook.

test-all:
##
# Notes:
# * Run all tests and set OMP/MKL threads 1 to allow pytest parallelization
# * The number of parallel tests will be 10 by default, which is good for running tests
#   locally. Larger numbers may cause OOM error. If PYTEST_WORKERS environment variable
#   is already set, the existing value gets used.
# * Currently, our tests fail when DDP is enabled. Hence, we set disable gpus for test
#   by setting CUDA_VISIBLE_DEVICES="".
##
	PYTEST_WORKERS="$${PYTEST_WORKERS:=10}"; CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest . -n $$PYTEST_WORKERS $(extra_args)

coverage-test-all:
	PYTEST_WORKERS="$${PYTEST_WORKERS:=10}"; CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 coverage run -m pytest . -n $$PYTEST_WORKERS $(extra_args)
