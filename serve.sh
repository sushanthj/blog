#!/bin/bash

case "$1" in
  "prod")
    echo "Starting production environment..."
    bundle exec jekyll serve --config _config.yml,_config.production.yml
    ;;
  "staging")
    echo "Starting staging environment..."
    bundle exec jekyll serve --config _config.yml,_config.staging.yml
    ;;
  *)
    echo "Starting development environment..."
    bundle exec jekyll serve
    ;;
esac 