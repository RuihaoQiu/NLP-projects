## Crawl names
## run: 'scrapy crawl name -o name.csv'

import scrapy
from scrapy.http import Request, FormRequest
from scrapy.contrib.spiders.init import InitSpider

## first name
class MovieSpider(scrapy.Spider):
  name = "name"
  allowed_domains = ['familyeducation.com']
  start_urls = ['https://www.familyeducation.com/baby-names/browse-origin/first-name/german']

  def parse(self, response):
    for name in response.css(".baby-names-list li"):
      yield {"Name": name.css("a::text").extract_first()}

    next_page = response.css('li.pager__item--next a::attr(href)').extract_first()
    if next_page is not None:
      yield response.follow(next_page, callback=self.parse)

"""
## family name
class MovieSpider(scrapy.Spider):
  name = "name"
  allowed_domains = ['familyeducation.com']
  start_urls = ['https://www.familyeducation.com/baby-names/browse-origin/surname/german']

  def parse(self, response):
    for name in response.css(".baby-names-list li"):
      yield {"Name": name.css("a::text").extract_first()}

    next_page = response.css('li.pager__item--next a::attr(href)').extract_first()
    if next_page is not None:
      yield response.follow(next_page, callback=self.parse)
"""
