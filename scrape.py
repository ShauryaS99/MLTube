#Imports
from rake_nltk import Rake
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException



#Extract Keywords
class Extract:
    def __init__(self, title):
        self.title = title
    def extract_keywords(self):
        r = Rake()
        r.extract_keywords_from_text(self.title)
        lst_scores = r.get_ranked_phrases_with_scores()
        keywords = [x[1] for x in lst_scores]
        return keywords


#Setup WebDriver


class Relevancy_Scraper:
    url = "https://www.youtube.com/results?search_query="
    def __init__(self, keyword):
        # GOOGLE_CHROME_PATH = '/ml-tube/.apt/usr/bin/google_chrome'
        # CHROMEDRIVER_PATH = '/ml-tube/.chromedriver/bin/chromedriver'
        # chrome_options = Options()
        # chrome_options.add_argument('--disable-gpu')
        # chrome_options.add_argument('--no-sandbox')
        # chrome_options.binary_location = GOOGLE_CHROME_PATH
        # driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=chrome_options)
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(executable_path='C:\\webdrivers\\chromedriver.exe', options=chrome_options)
        self.keyword = keyword
        self.driver = driver
        self.livestream = False
        self.lst_views = []
        self.lst_dates = []
    def modify_url(self):
        lst_keywords = self.keyword.split(" ")
        query = ""
        for word in lst_keywords:
            query+=word + "+"
        query = query[:-1]
        Relevancy_Scraper.url+=query
    def scrape(self):
        self.driver.get(Relevancy_Scraper.url)
        lst_views = []
        lst_dates = []
        counter = 0
        for i in range(1, 10):
            try:
                xpath = "/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{}]/div[1]/div/div[1]/ytd-video-meta-block/div[1]/div[2]/span[1]".format(i)
                curr_views = self.driver.find_element_by_xpath(xpath).text
                # print(curr_views)
                if "watching" not in curr_views:
                    dates_xpath = "/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{}]/div[1]/div/div[1]/ytd-video-meta-block/div[1]/div[2]/span[2]".format(i)
                    curr_dates = self.driver.find_element_by_xpath(dates_xpath).text
                    lst_dates.append(curr_dates)
                    lst_views.append(curr_views)
                    counter+=1
                    # print(counter)
                else:
                    self.livestream = True
                    continue
                if counter >=5:
                    break
            except NoSuchElementException:
                pass
        self.lst_views = lst_views
        self.lst_dates  = lst_dates
    
    # total_views = sum(lst_)
    def convert_str_to_number(self, views):
        x = views.split(" ")[0]
        number = 0
        num_map = {'K':1000, 'M':1000000, 'B':1000000000}
        if x.isdigit():
            number = int(x)
        else:
            if len(x) > 1:
                number = float(x[:-1]) * num_map.get(x[-1].upper())
        return int(number)

    # total_views = sum(lst_)
    def convert_str_to_date(self, date):
        x = date.split(" ")
        time_period = x[1]
        if "month" in time_period:
            return int(x[0]) * 1.0
        elif "year" in time_period:
            return int(x[0]) * 12.0
        else:
            return int(x[0]) / 100.0


    def adjust(self, lst_dates, lst_views):
        decay_rate = 0.35
        adjusted_dates = []
        for date in lst_dates:
            adjusted_dates.append(self.convert_str_to_date(date))
        adjusted_views = []
        # print(adjusted_dates)
        # print(lst_views)
        for i in range(3):            
            int_view = self.convert_str_to_number(lst_views[i]) 
            if adjusted_dates[i] < 1:
                hours = adjusted_dates[i] * 100
                views_at_time = int_view
                rate = views_at_time / hours
                adj_view = 24 * rate
            else:
                adj_view = int_view * (decay_rate**adjusted_dates[i])
            adjusted_views.append(adj_view)
        return adjusted_views
    
    def judgement(self, tot_views):
        if tot_views < 10000:
            return "Irrelevant"
        elif tot_views < 100000:
            return "Niche"
        elif tot_views < 1000000:
            return "Relevant"
        else:
            return "Popular"
    
    def get_adj_views(self):
        self.modify_url()
        self.scrape()
        adjusted_views = self.adjust(self.lst_dates, self.lst_views)
        sum_views = sum(adjusted_views)
        return round(sum_views, 2)
    
    def close(self):
        self.driver.quit()

    def to_string(self, total_views):
        livestream = "" if self.livestream else "not "
        relevancy = self.judgement(total_views)
        keyword = self.keyword
        result = f"The topic {keyword} has been classified as {relevancy} with about {total_views} views. There are {livestream}currently livestreams on this topic."
        return result
