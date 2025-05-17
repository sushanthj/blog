module Jekyll
  module ImagePathFilter
    def self.add_baseurl_to_images(content)
      return content unless content.is_a?(String)
      
      # Match markdown image syntax for any path starting with /
      content = content.gsub(/(!\[.*?\]\()(\/[^)]*\))/, "\\1{{ site.baseurl }}\\2")
      
      # Match HTML img tags with src="/..."
      content = content.gsub(/(<img[^>]*src=")(\/[^"]*")/, "\\1{{ site.baseurl }}\\2")
      
      # Match HTML img tags with src='/...'
      content = content.gsub(/(<img[^>]*src=')(\/[^']*')/, "\\1{{ site.baseurl }}\\2")
      
      # Match background-image: url('/...')
      content = content.gsub(/(background-image:\s*url\()(\/[^)]*\))/, "\\1{{ site.baseurl }}\\2")
      
      # Match background-image: url("/...")
      content = content.gsub(/(background-image:\s*url\(")(\/[^"]*"\))/, "\\1{{ site.baseurl }}\\2")
      
      content
    end
  end
end

# Add a hook to process markdown files
Jekyll::Hooks.register [:posts, :pages, :documents], :pre_render do |doc|
  doc.content = Jekyll::ImagePathFilter.add_baseurl_to_images(doc.content)
end 