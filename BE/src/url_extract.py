import yake
import streamlit as st
import article_crawl as ac
from flashtext import KeywordProcessor
from collections import Counter

DEFAULT_KEYWORDS = ["mới nhất", "hiện đại nhất", "độc quyền", "duy nhất", "hoàn toàn", "nhất", "hoàn toàn 100%"]

def main(): 
    # st.set_page_config(page_title="Demo Lọc Từ khóa Vi phạm", page_icon="🔍")
    st.title("Demo Lọc từ khóa vi phạm từ url")

    url = st.text_input("Nhập URL", placeholder="https://www.example.com")    
    filter_btn = st.button(label="Lọc", use_container_width=True)
    if filter_btn:
        show_result(url=url, keywords=DEFAULT_KEYWORDS)

def extract_keywords(article: str, keywords: list[str]) -> dict[str, int]:
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(keywords)
    extracted_keywords = keyword_processor.extract_keywords(article)
    
    # Count the frequency of each keyword
    keyword_frequency = dict(Counter(extracted_keywords))
    
    return keyword_frequency

def show_result(url:str, keywords: list[str]):
    # Kiểm tra nếu URL và từ khóa không được nhập
    if not url:
        st.error("Vui lòng nhập đầy đủ thông tin.")
    else:
        article = ac.crawl_and_clean_article(url=url)

        filtered_keywords = extract_keywords(article["content"], keywords)
        sentences_with_keywords = get_sentences_with_keywords(article, filtered_keywords.keys())

        if len(filtered_keywords) == 0:
            st.error("Không tìm thấy từ khóa trong bài viết.")
        else:
            # st.subheader("Danh sách câu có chứa từ khóa vi phạm")
            st.text_area("Danh sách câu có chứa từ khóa vi phạm","- " + "\n- ".join([f"{s}" for s in sentences_with_keywords]), height=300)
            st.success("Kết quả lọc <từ khóa>: <số lần xuất hiện>:\n- " + "\n- ".join([f"{kw}: {f}" for kw, f in filtered_keywords.items()]))

        # st.success(f"Từ khóa khác trong bài viết - Yake: \n- " + "\n- ".join(topYake(article, DEFAULT_KEYWORDS.lower().split(", "), filtered=filtered_options, length=length_keywords_list)))

def get_sentences_with_keywords(article: ac._article, keywords: list[str]) -> list[str]:
    sentences = article["content"].split(". ")
    sentences_with_keywords = []
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            sentences_with_keywords.append(sentence)
    return list(set(sentences_with_keywords))



# def topYake(article, word_list:list[str], length:int=10, filtered:bool=False):
#     kw_extractor = yake.KeywordExtractor(lan="vi", top=100)
#     article_keywords = kw_extractor.extract_keywords(article["content"] + ". " + article["title"])
#     sorted_article_keywords = sorted(article_keywords, key=lambda x: x[1], reverse=False)

#     if not filtered:
#         top = [kw for kw, score in sorted_article_keywords]
#         return top[:length]
#     filtered_keywords = [kw for kw, score in sorted_article_keywords if any(word.lower() in kw.lower() for word in word_list)]
#     return filtered_keywords[:length]

# def top10KeyBert(article):
#     kw_extractor = KeyBERT()
#     article_keywords = kw_extractor.extract_keywords((article["title"] + " " + article["content"]).lower(), keyphrase_ngram_range=(1, 3), stop_words='vietnamese', top_n=10)
#     top10 = [kw for kw, score in article_keywords]
#     return top10

main()
