# Make is verbose in Linux. Make it silent.
MAKEFLAGS += --silent

pre-push-hook:
	cp pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
.PHONY: pre-push-hook