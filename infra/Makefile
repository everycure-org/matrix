
.PHONY: secrets
secrets:
	git submodule update --recursive --init
	git submodule update --recursive --remote
	cd secrets && git-crypt unlock

init:
	terragrunt run-all init

plan: secrets
	terragrunt run-all plan

