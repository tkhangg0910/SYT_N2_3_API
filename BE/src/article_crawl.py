import os
import csv
from newspaper import Article

class _article:
    title: str
    content: str
    gpkd: str

    def __init__(self, title:str, content:str):
        self.title = title
        self.content = content

def crawl_article(url) -> _article:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {"title":article.title, "content":article.text}
    except Exception as e:
        print(f"Error crawling article: {e}")

def clean_content(text:str) -> str:
    # Remove extra spaces and newlines
    text = text.replace('\n\n', '. ').replace('\n', ' ').replace('\t', ' ').replace('- ', ' ').replace('..', '.')
    text = ' '.join(text.split())
    return text

def crawl_articles(url_list:list):
    articles = []
    for url in url_list:
        articles.append(crawl_article(url))
    return articles

def clean_articles(articles:list) -> list[_article]:
    cleaned_articles = []
    for article in articles:
        cleaned_article = {"title":article["title"], "content":clean_content(article["content"])}
        cleaned_articles.append(cleaned_article)
    return cleaned_articles

def crawl_and_clean_article(url:str) -> _article:
    article = crawl_article(url)
    cleaned_article = clean_content(article["content"])
    article["content"] = cleaned_article
    return article

def crawl_and_clean_articles(url_list:list) -> list[_article]:
    articles = crawl_articles(url_list)
    cleaned_articles = clean_articles(articles)
    return cleaned_articles

def output(articles:list, title_as_filename:bool=True, csv:bool=False):
    if not csv:
        __output_file(articles, title_as_filename=title_as_filename)
    else:
        __output_csv(articles)

def __output_file(articles:list, title_as_filename:bool=True):
    # Check if any articles were found
    if (len(articles) == 0):
        print("Error: No articles found.")
        return
    
    # Check if the output directory exists to delete, if not create it
    out_path = "res"
    if (os.path.exists(out_path) == True):
        for file in os.listdir(out_path):
            os.remove(os.path.join(out_path, file))
        os.rmdir(out_path)
    os.makedirs(out_path)


    # Create a directory for each article and save the content to a file
    if(title_as_filename):
        for article in articles:
            with open(os.path.join(out_path, f"{article['title']}.txt"), "w") as f:
                f.write(article["content"])
                print(f"Article {article['title']} saved to {out_path}")
    else:
        counter = 1
        for article in articles:
            with open(os.path.join(out_path, f"{counter}.txt"), "w") as f:
                f.write(article["content"])
                print(f"Article {article['title']} saved to {out_path}")
                counter += 1

def __output_csv(articles:list):
    with open('res.csv', 'w', newline='') as csvfile:
        fieldnames = ['title', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(articles)
    print("CSV file created successfully")

if __name__ == "__main__":
    urls = ["https://thammyviensline.com/tham-vien-sline-su-lua-chon-hoan-hao-cua-ban/", "https://plo.vn/csgt-thong-tin-ban-dau-vu-o-to-khach-bi-lat-o-phu-yen-post833337.html", "https://tapchigiaothong.vn/rao-ban-vinfast-vf3-sau-10-km-nguoi-dung-noi-thang-mot-dieu-183250207203953709.htm", "https://vov.vn/the-gioi/toretsk-ruc-lua-ukraine-tuyen-bo-day-lui-moi-cuoc-tan-cong-cua-nga-tai-day-post1153466.vov"]
    output(crawl_and_clean_articles(urls), title_as_filename=False, csv=False)
