# Keep in sync with .ruby-version (compose passes this as a build arg).
ARG RUBY_VERSION=3.1
FROM ruby:${RUBY_VERSION}-slim

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  nodejs \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy Gemfile and install gems. This is layered to use Docker's build cache.
COPY Gemfile Gemfile.lock ./
RUN gem install bundler && \
    bundle install

# The site itself is bind-mounted at /app by docker-compose, so it is deliberately
# not COPYed here — that would be shadowed at runtime and would bust the build
# cache on every content edit.