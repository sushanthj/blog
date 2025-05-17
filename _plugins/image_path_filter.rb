module Jekyll
  module ImagePathFilter
    def self.add_baseurl_to_images(content)
      return content unless content.is_a?(String)
      
      # More permissive regex that will catch any markdown image syntax
      # including those with or without alt text, and with or without leading slash
      content.gsub(/!\[(.*?)\]\((.*?)\)/) do |match|
        alt_text = $1
        path = $2
        
        # Only add baseurl if the path starts with a slash
        if path.start_with?('/')
          "![#{alt_text}]({{ site.baseurl }}{{ page.url | replace: '.html', '' }}/#{path[1..-1]})"
        else
          match
        end
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
  if doc.respond_to?(:content) && doc.content.is_a?(String)
    doc.content = Jekyll::ImagePathFilter.add_baseurl_to_images(doc.content)
  end
end 