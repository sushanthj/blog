module Jekyll
    module ImagePathFilter
      def add_baseurl_to_images(content)
        return content unless content.is_a?(String)
  
        content.gsub(/!\[(.*?)\]\((\/.*?)\)/) do
          alt_text = Regexp.last_match(1)
          path = Regexp.last_match(2)
          # Use a Liquid template that defers resolution to Jekyll
          "![#{alt_text}]({% raw %}{{ site.baseurl }}{{ page.url | replace: '.html', '' }}#{path}{% endraw %})"
        end
      end
    end
  end
  
  Liquid::Template.register_filter(Jekyll::ImagePathFilter)
  
  # Hook to apply the filter before rendering
  Jekyll::Hooks.register [:pages, :posts, :documents], :pre_render do |doc|
    next unless doc.output_ext == '.html' && doc.content.is_a?(String)
  
    # Apply the transformation before rendering
    filter = Jekyll::ImagePathFilter.new
    doc.content = filter.add_baseurl_to_images(doc.content)
  end
  