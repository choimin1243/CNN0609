from bs4 import BeautifulSoup as bs
from urllib.request import urlopen



years = list(range(2010, 2019))
monthes = list(range(1, 13))
days = list(range(0, 5))

rating_pages = []

for year in years:
    for month in monthes:
        for day in days:
            respone=urlopen("https://workey.codeit.kr/ratings/index?year="+str(year)+"&month="+str(month)+"&weekIndex="+str(day))
            soup=bs(respone,'html.parser')
            a=soup.select(".row .channel")
            for li in a:
                rating_pages.append(li.text)



print(rating_pages)

