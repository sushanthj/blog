FROM ruby:3.1-slim

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libgl1-mesa-glx \
  nodejs \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Add Gemfile and Gemfile.lock to the container
COPY Gemfile ./

# Install python3, pip, and PyQt5
RUN apt-get update && apt-get install -y python3 python3-pip python3-pyqt5

# Install gems from Gemfile
RUN gem install bundler
RUN bundle install

# Generate Gemfile.lock
RUN bundle lock

# Copy the Jekyll site to the container
COPY . .