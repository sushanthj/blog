module Jekyll
  module ImagePathFilter
    def self.add_baseurl_to_images(content)
      return content unless content.is_a?(String)
      
      # Handle both ![alt](path) and ![](path) syntax
      content.gsub(/(!\[(.*?)\]\((\/[^)]*)\))/) do |match|
        "![#{$2}]({{ site.baseurl }}#{$3})"
      end
    end
  end
end

# Register as a filter that can be used in templates
module Jekyll
  module Filters
    def add_baseurl_to_images(input)
      Jekyll::ImagePathFilter.add_baseurl_to_images(input)
    end
  end
end

# Add a hook to process markdown files
Jekyll::Hooks.register [:posts, :pages, :documents], :pre_render do |doc|
  doc.content = Jekyll::ImagePathFilter.add_baseurl_to_images(doc.content)
end 