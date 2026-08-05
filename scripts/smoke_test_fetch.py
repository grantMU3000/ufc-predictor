from data.scraping.fetch import fetch

html = fetch("http://ufcstats.com/statistics/events/completed")
print(len(html))
