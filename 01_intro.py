import scrapy


class MySpider(scrapy.Spider):
    name = 'huffingtonpost'
    allowed_domains = ['huffingtonpost.com/']
    start_urls = [
        'https://www.huffingtonpost.com/section/politics',
        'https://www.huffingtonpost.com/dept/entertainment',
        'https://www.huffingtonpost.com/section/media',
    ]

    def parse(self, response):
        for h3 in response.xpath('//h3').extract():
            yield {"title": h3}

        for url in response.xpath('//a/@href').extract():
            if url.startswith('/'):
                # transform url into absolute
                url = 'http://www.huffingtonpost.com' + url
            if url.startswith('#'):
                # ignore href starts with #
                continue
            yield scrapy.Request(url, callback=self.parse)