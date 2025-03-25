# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal blog/portfolio site at www.sush.one, built with Jekyll using the Duet theme. Hosted on GitHub Pages.

## Development Commands

```bash
# Install dependencies
bundle install

# Local dev server (default)
bundle exec jekyll serve

# Production config
bundle exec jekyll serve --config _config.yml,_config.production.yml

# Staging config
bundle exec jekyll serve --config _config.yml,_config.staging.yml

# GUI tool (via Docker)
docker compose up
```

There's also `gui/serve.sh [prod|staging]` as a shortcut for the serve commands.

## Architecture

- **Jekyll site** using `github-pages` gem (not standalone Jekyll)
- **Three collections**: `_posts/`, `_pages/`, and projects (configured in `_config.yml`)
- Posts and pages are organized into subdirectories by category (art, food, projects, knowledge-base, work)
- **Config layering**: `_config.yml` (base) + `_config.production.yml` or `_config.staging.yml` for environment overrides
- **Theme settings**: `_data/settings.yml` controls colors, fonts, navigation, social links, contact form
- **Custom plugin**: `_plugins/image_path_filter.rb`
- **GUI tool**: `gui/gui.py` — a PyQt5 desktop app run via Docker (`docker-compose.yaml`)
- **Layouts**: `_layouts/` (default, page, post) with partials in `_includes/`
- **Styles**: `_sass/` compiled via Jekyll's asset pipeline
