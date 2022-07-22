# Make is verbose in Linux. Make it silent.
MAKEFLAGS += --silent

# If the first argument is "run"...
ifeq (commit,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

prog: # ...
    # ...

test: prog
	@echo prog $(RUN_ARGS)
.PHONY: test


commit: prog
	@echo $$(( $$(cat ./version) + 1 )) > version
  git add ./version
	git commit -m "$(RUN_ARGS)"
.PHONY: commit