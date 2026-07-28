PORT ?= 4100
RUBY_VERSION := $(shell cat .ruby-version)
COMPOSE := docker compose -f docker/docker-compose.yaml
DOCKER_UID := $(shell id -u)
DOCKER_GID := $(shell id -g)
export PORT RUBY_VERSION DOCKER_UID DOCKER_GID

.DEFAULT_GOAL := help
.PHONY: help serve staging prod build shell clean

help: ## Show this help
	@echo "Blog — www.sush.one   (everything runs in Docker)"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-8s %s\n", $$1, $$2}'
	@echo
	@echo "  Serves on http://localhost:$(PORT) — override with PORT=4200"

# serve/staging pass --unpublished so `published: false` drafts are visible locally.
# prod deliberately omits it, so it renders exactly what the live site will.
serve: ## Start the dev server (drafts visible)
	JEKYLL_CONFIG=_config.yml JEKYLL_FLAGS=--unpublished $(COMPOSE) up --build

staging: ## Start the dev server with the staging config (drafts visible)
	JEKYLL_CONFIG=_config.yml,config/staging.yml JEKYLL_FLAGS=--unpublished $(COMPOSE) up --build

prod: ## Start the dev server with the production config (drafts hidden)
	JEKYLL_CONFIG=_config.yml,config/production.yml $(COMPOSE) up --build

build: ## Build the static site into _site/
	$(COMPOSE) run --rm --no-deps web bundle exec jekyll build

shell: ## Open a shell in the container
	$(COMPOSE) run --rm --no-deps web bash

clean: ## Remove the generated site and containers
	# Runs as root inside the container so it can also clear output left over
	# from before the `user:` mapping was added.
	$(COMPOSE) run --rm --no-deps --user 0:0 web rm -rf /app/_site /app/.jekyll-cache
	$(COMPOSE) down --remove-orphans
