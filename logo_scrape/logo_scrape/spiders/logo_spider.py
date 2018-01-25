from logo_scrape.items import LogoScrapeItem
import scrapy
import re
import csv

def get_data():
    path = 'logo_scrape/static/data.csv'
    output_list = list()
    with open(path) as f:
        data_read = csv.reader(f, delimiter='\n')
        for row in data_read:
            el_http = "http://" + str(row[0])
            output_list.append(el_http)
            # DO I NEED THIS:
            # el_https = "https://" + str(row[0])
            # output_list.append(el_https)
    return output_list


class LogoSpider(scrapy.Spider):
    name = "logo_spider"

    def start_requests(self):
        self.site_id = 0
        for el in get_data():
            yield scrapy.Request(el, self.parse)
        #yield scrapy.Request("https://time.com/", self.parse)

    def parse(self, response):
        self.site_id += 1
        self.link_id = response.url
        self.status = -9999
        self.current_link = ""
        self.link_img = ""
        self.pattern_id = -9999
        self.found_image = False

        pattern_regex = r"<[0-9][0-9][0-9]\W"
        response_status_code = int(re.match(pattern_regex,
                                        str(response)).group(0)[1:4])
        self.status = response_status_code
        # TODO TEST THIS
        # Here we handled bad requests
        # Tova ne ba4ka kakto trqbva, moje bi trqbva da se izpolzva druga biblioteka da proverqva predvaritelno
        if response_status_code != 200:
            yield LogoScrapeItem(site_id = self.site_id,
                    link_id=self.link_id,
                    status = self.status,
                    current_link = self.current_link,
                    pattern_id = self.pattern_id,
                    found_img = self.found_image,
                    link_img=self.link_img
                    )
            
        #Here we get 10 random pages for analysing
        iterated_links_counter = 0
        for href in response.xpath("//a"):
            if iterated_links_counter >= 10:
                break
            valid_url_pattern = r"https?://.+"
            link_to_next_page = href.xpath("@href").extract_first()
            if not re.match(valid_url_pattern, str(link_to_next_page)):
                continue    
            iterated_links_counter += 1
            yield scrapy.Request(link_to_next_page, self.parse_inner_page)

    def parse_inner_page(self, response):
        self.current_link = response.url
        xpath_counter = 0
        xpath_img_patterns = [response.css("#Header").xpath("//a/img/@src"), 
                              response.css("#HEADER").xpath("//a/img/@src"),    
                              response.css("#header").xpath("//a/img/@src"),
                              response.xpath("//header//a/img/@src"),
                              response.xpath("//a[@href='"+response.url+'/'+"']/img/@src"),
                              response.xpath("//a[@href='"+response.url+'/'+"']/img/@src"),
                              response.css("header").xpath("//a[@href='/']/img/@src"),
                              response.xpath("//a[@href='/']/img/@src")
                              ]
        for xpath_pattern in xpath_img_patterns:
            xpath_counter += 1
            if xpath_pattern:
                self.pattern_id = xpath_counter
                self.found_img = True
                self.link_img = xpath_pattern.extract_first()
                yield LogoScrapeItem(site_id = self.site_id,
                    link_id=self.link_id,
                    status = self.status,
                    current_link = self.current_link,
                    pattern_id = self.pattern_id,
                    found_img = self.found_image,
                    link_img = self.link_img
                    )

