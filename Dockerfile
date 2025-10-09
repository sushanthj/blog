FROM ruby:3.1-slim

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libgl1-mesa-glx \
  nodejs \
  python3 \
  python3-pip \
  python3-pyqt5 \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy Gemfile and install gems. This is layered to use Docker's build cache.
COPY Gemfile ./
RUN gem install bundler && \
    bundle install && \
    bundle lock

# Copy the Jekyll site to the container
COPY . .