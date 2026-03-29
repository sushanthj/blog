FROM ruby:3.1-slim

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

# Copy the Jekyll site to the container
COPY . .