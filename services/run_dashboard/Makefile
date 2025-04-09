# TODO: add image name etc when it comes to deploying

.PHONY: clean install set_run_id run-dev

clean:
	rm -rf .evidence
	rm -rf node_modules
	rm -f package-lock.json

install:
	npm install

set_run_id:
	node ./scripts/set-run-id.cjs $(RUN_ID)

build: set_run_id
	npm run build

run-dev: set_run_id
	npm run sources
	npm run dev
