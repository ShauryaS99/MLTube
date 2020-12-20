#Imports
from rake_nltk import Rake
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import os

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
    def __init__(self, keyword):
        GOOGLE_CHROME_PATH = '/app/.apt/usr/bin/google_chrome'
        CHROMEDRIVER_PATH = '/app/.chromedriver/bin/chromedriver'
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
        driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), options=chrome_options)
        # driver = webdriver.Chrome(executable_path='C:\\webdrivers\\chromedriver.exe', options=chrome_options)
        self.url = "https://www.youtube.com/results?search_query="
        self.keyword = keyword
        self.driver = driver
        self.livestream = False
        self.lst_views = []
        self.lst_dates = []
        self.shit_failed = False
    def modify_url(self):
        lst_keywords = self.keyword.split(" ")
        query = ""
        for word in lst_keywords:
            query+=word + "+"
        query = query[:-1]
        self.url+=query
        print("keywords are: ")
        print(lst_keywords)
    def scrape(self):
        self.driver.get(self.url)
        filtered = False
        lst_views = []
        lst_dates = []
        counter = 0
        filter_btn = "/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[1]/div[2]/ytd-search-sub-menu-renderer/div[1]/div/ytd-toggle-button-renderer/a"
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, filter_btn))).click()
        filter_month = "/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[1]/div[2]/ytd-search-sub-menu-renderer/div[1]/iron-collapse/div/ytd-search-filter-group-renderer[1]/ytd-search-filter-renderer[4]/a"
        WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.XPATH, filter_month))).click()
        time.sleep(3)
        for i in range(1, 10):
            try:
                xpath = "/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{}]/div[1]/div/div[1]/ytd-video-meta-block/div[1]/div[2]/span[1]".format(i)
                curr_views = self.driver.find_element_by_xpath(xpath).text
                print(curr_views)
                if "watching" not in curr_views:
                    dates_xpath = "/html/body/ytd-app/div/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{}]/div[1]/div/div[1]/ytd-video-meta-block/div[1]/div[2]/span[2]".format(i)
                    curr_dates = self.driver.find_element_by_xpath(dates_xpath).text
                    lst_dates.append(curr_dates)
                    lst_views.append(curr_views)
                    counter+=1
                    print(counter)
                else:
                    self.livestream = True
                    continue
                if counter >=5:
                    break
            except NoSuchElementException:
                pass
        self.lst_views = lst_views
        self.lst_dates  = lst_dates
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
        try:
            self.modify_url()
            self.scrape()
            adjusted_views = self.adjust(self.lst_dates, self.lst_views)
            sum_views = sum(adjusted_views)
            return round(sum_views, 2)
        except:
            self.shit_failed = True
            return 0
    
    def close(self):
        self.driver.quit()
        
    def to_string(self, total_views):
        keyword = self.keyword
        if self.shit_failed:
            return f"Unable to calculate relevancy for topic {keyword}. Try modifying the title or trying again."
        livestream = "" if self.livestream else "not "
        relevancy = self.judgement(total_views)
        result = f"The topic {keyword} has been classified as {relevancy} with about {total_views} views. There are {livestream}livestreams on this topic."
        return result