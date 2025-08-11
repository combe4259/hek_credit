const puppeteer = require("puppeteer");
const { MongoClient } = require("mongodb");

// MongoDB connection
const uri = "mongodb+srv://julk0206:%23Sooyeon2004@hek.yqi7d9x.mongodb.net";
const client = new MongoClient(uri);

// 네이버 뉴스 통합뷰 URL로 변환
function convertToNaverNewsView(url) {
    if (url.includes('n.news.naver.com')) {
        return url;
    }
    
    const oidMatch = url.match(/[?&]oid=(\d+)/);
    const aidMatch = url.match(/[?&]aid=(\d+)/);
    
    if (oidMatch && aidMatch) {
        const oid = oidMatch[1];
        const aid = aidMatch[1];
        return `https://n.news.naver.com/mnews/article/${oid}/${aid}`;
    }
    
    return url;
}

// 개선된 본문 추출 함수
async function getDetailContent(browser, url) {
    let detailPage;
    try {
        detailPage = await browser.newPage();
        await detailPage.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');
        
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const naverViewUrl = convertToNaverNewsView(url);
        console.log(`원본 URL: ${url}`);
        console.log(`통합뷰 URL: ${naverViewUrl}`);
        
        await detailPage.goto(naverViewUrl, {
            waitUntil: 'networkidle2',
            timeout: 60000
        });

        // 네이버 뉴스 통합뷰 전용 셀렉터
        const naverNewsSelectors = [
            '#dic_area',
            '#newsct_article',
            '.news_end_body',
            '._article_body_contents'
        ];
        
        // 언론사별 셀렉터
        const pressSelectors = [
            '.article_body', '.news_contents', '.article_content', '.news_content',
            '.end_contents_body', '#articleBodyContents', '#articeBody',
            '.news_view', '.article_txt', '.article_view', '.view_txt',
            '.read_body', '.article_area', '.news_article', '.content_area',
            '#content', '.content', '.post_content', '.entry_content',
            'article', 'main', '.main'
        ];

        // 네이버 통합뷰 시도
        for (const selector of naverNewsSelectors) {
            const hasContent = await detailPage.$(selector);
            if (hasContent) {
                const content = await detailPage.$eval(selector, el => el.innerText.trim());
                if (content && content.length > 100) {
                    console.log(`✅ 네이버 통합뷰에서 본문 추출 성공 (${selector}) - ${content.length}자`);
                    return content;
                }
            }
        }
        
        // 언론사 사이트 시도
        console.log('⚠️ 네이버 통합뷰에서 본문을 찾지 못함. 언론사 사이트 시도...');
        for (const selector of pressSelectors) {
            try {
                const hasContent = await detailPage.$(selector);
                if (hasContent) {
                    const content = await detailPage.$eval(selector, el => {
                        let text = el.innerText || el.textContent || '';
                        const elementsToRemove = el.querySelectorAll('script, style, .ad, .advertisement, .related, .share, .comment, nav, header, footer');
                        elementsToRemove.forEach(elem => elem.remove());
                        text = el.innerText || el.textContent || text;
                        return text.trim();
                    });
                    
                    if (content && content.length > 100) {
                        console.log(`✅ 언론사 사이트에서 본문 추출 성공 (${selector}) - ${content.length}자`);
                        return content;
                    }
                }
            } catch (selectorError) {
                continue;
            }
        }
        
        return "본문을 가져올 수 없습니다.";
    } catch (error) {
        console.log(`상세 내용 크롤링 실패: ${url} - ${error.message}`);
        return "본문을 가져올 수 없습니다.";
    } finally {
        if (detailPage) {
            await detailPage.close();
        }
    }
}

async function testSingleArticle() {
    let browser;
    try {
        await client.connect();
        const db = client.db("newsDB");
        const collection = db.collection("news");
        
        // 테스트할 URL들 (실제 수집된 기사 중에서)
        const testUrls = [
            "https://www.newsis.com/view/NISX20250728_0003269632",
            "https://www.joongang.co.kr/article/25353144",
            "https://it.chosun.com/news/articleView.html?idxno=2023092144449"
        ];
        
        browser = await puppeteer.launch({ headless: true });
        
        for (const url of testUrls) {
            console.log(`\n=== 테스트 URL: ${url} ===`);
            const content = await getDetailContent(browser, url);
            console.log(`본문 길이: ${content.length}자`);
            console.log("본문 내용:");
            console.log("=" + "=".repeat(100));
            console.log(content);
            console.log("=" + "=".repeat(100));
            console.log("\n");
        }
        
    } catch (error) {
        console.error("테스트 중 오류:", error);
    } finally {
        if (browser) await browser.close();
        await client.close();
    }
}

testSingleArticle();