const puppeteer = require("puppeteer");
const cheerio = require("cheerio"); // Cheerio 모듈 추가
const { MongoClient } = require("mongodb");
const https = require('https'); // 이 모듈은 현재 직접 사용되지 않지만, 기존 코드에 있어 남겨둡니다.
const fs = require('fs');
const iconv = require('iconv-lite'); // EUC-KR 인코딩용

// 한글을 EUC-KR로 인코딩하는 함수
function encodeToEUCKR(koreanText) {
    try {
        const eucKrBuffer = iconv.encode(koreanText, 'euc-kr');
        let encoded = '';
        for (let i = 0; i < eucKrBuffer.length; i++) {
            encoded += '%' + eucKrBuffer[i].toString(16).toUpperCase().padStart(2, '0');
        }
        return encoded;
    } catch (error) {
        console.error(`EUC-KR 인코딩 실패 (${koreanText}):`, error.message);
        // fallback으로 UTF-8 사용
        return encodeURIComponent(koreanText);
    }
}

// fetchWithRetry 함수 (Puppeteer 기반)
async function fetchWithRetry(page, url, maxRetries = 3, delay = 2000) {
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            if (attempt > 1) {
                const jitter = Math.random() * 1000;
                await new Promise(resolve => setTimeout(resolve, delay + jitter));
            }

            console.log(`Attempt ${attempt} for ${url} with Puppeteer`);

            const response = await page.goto(url, {
                waitUntil: 'networkidle2', // 네트워크 활동이 2초 이상 없을 때까지 대기
                timeout: 60000 // 타임아웃을 1분으로 늘려 안정성 향상
            });

            if (response.status() === 200) {
                return response;
            } else if (response.status() === 403) {
                console.warn(`Access denied (403) for ${url}. Retrying...`);
                delay += 2000;
            } else if (response.status() === 429) {
                console.warn(`Rate limited (429) for ${url}. Waiting before retry...`);
                await new Promise(resolve => setTimeout(resolve, 10000));
            } else {
                console.warn(`Unexpected status ${response.status()} for ${url}`);
            }
        } catch (error) {
            lastError = error;
            console.warn(`Request failed (${error.message}). Retrying...`);
            if (error.message.includes('timeout') && delay < 15000) {
                delay += 5000;
            }
        }
    }

    throw lastError || new Error(`Failed to fetch ${url} after ${maxRetries} attempts with Puppeteer`);
}

// MongoDB 연결 설정 (환경변수 또는 설정 파일에서 가져오기)
const uri = process.env.MONGODB_URI || "mongodb+srv://julk0206:%23Sooyeon2004@hek.yqi7d9x.mongodb.net";
const client = new MongoClient(uri);

// Korean date time + relative date time parsing
function parseKoreanDate(dateStr) {
    try {
        const now = new Date();

        if (dateStr.includes('분 전')) {
            const minutes = parseInt(dateStr.match(/(\d+)분 전/)?.[1] || 0);
            return new Date(now.getTime() - minutes * 60 * 1000);
        }

        if (dateStr.includes('시간 전')) {
            const hours = parseInt(dateStr.match(/(\d+)시간 전/)?.[1] || 0);
            return new Date(now.getTime() - hours * 60 * 60 * 1000);
        }

        if (dateStr.includes('일 전')) {
            const days = parseInt(dateStr.match(/(\d+)일 전/)?.[1] || 0);
            return new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
        }

        const dateMatch = dateStr.match(/(\d{4})\.(\d{2})\.(\d{2})\./); // YYYY.MM.DD.
        if (dateMatch) {
            return new Date(parseInt(dateMatch[1]), parseInt(dateMatch[2]) - 1, parseInt(dateMatch[3]));
        }

        return now;
    } catch (e) {
        console.error(`Error parsing date string "${dateStr}":`, e.message);
        return new Date();
    }
}

// Check for duplicate articles in MongoDB
async function isDuplicate(collection, title, url) {
    const existing = await collection.findOne({
        $or: [
            { title: title },
            { url: url }
        ]
    });

    if (existing) {
        console.log(`중복 상세정보:`);
        console.log(`     - 새 기사: "${title}"`);
        console.log(`     - 기존 기사: "${existing.title}"`);
        console.log(`     - 새 URL: ${url}`);
        console.log(`     - 기존 URL: ${existing.url}`);
        if (title === existing.title) {
            console.log(`     - 제목이 동일함`);
        }
        if (url === existing.url) {
            console.log(`     - URL이 동일함`);
        }
    }

    return !!existing;
}

// 네이버 뉴스 통합뷰 URL로 변환
function convertToNaverNewsView(url) {
    // 이미 네이버 뉴스 통합뷰 URL인 경우
    if (url.includes('n.news.naver.com')) {
        return url;
    }

    // 네이버 뉴스 링크에서 oid와 aid 추출
    const oidMatch = url.match(/[?&]oid=(\d+)/);
    const aidMatch = url.match(/[?&]aid=(\d+)/);

    if (oidMatch && aidMatch) {
        const oid = oidMatch[1];
        const aid = aidMatch[1];
        // 네이버 뉴스 통합뷰 URL로 변환
        return `https://n.news.naver.com/mnews/article/${oid}/${aid}`;
    }

    // 변환할 수 없는 경우 원본 URL 반환
    return url;
}

// 뉴스 디테일 정보 크롤링 (제목, 언론사, 날짜, 본문 모두 포함)
async function getDetailContent(browser, url) {
    let detailPage;
    try {
        detailPage = await browser.newPage();
        await detailPage.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');

        await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));

        // 네이버 뉴스 통합뷰 URL로 변환 시도
        const naverViewUrl = convertToNaverNewsView(url);
        console.log(`원본 URL: ${url}`);
        console.log(`통합뷰 URL: ${naverViewUrl}`);

        await detailPage.goto(naverViewUrl, {
            waitUntil: 'networkidle2',
            timeout: 60000
        });

        // 네이버 뉴스 통합뷰 전용 셀렉터
        const naverNewsSelectors = [
            '#dic_area',           // 네이버 뉴스 메인 본문
            '#newsct_article',     // 네이버 뉴스 본문 (신버전)
            '.news_end_body',      // 네이버 뉴스 본문 (또 다른 버전)
            '._article_body_contents' // 네이버 뉴스 본문 (구버전)
        ];

        // 언론사별 셀렉터 (폴백용) - 2025년 업데이트된 다양한 언론사 셀렉터
        const pressSelectors = [
            // 일반적인 본문 셀렉터
            '.article_body', '.news_contents', '.article_content', '.news_content',
            '.end_contents_body', '#articleBodyContents', '#articeBody',
            '.go_trans._article_content', '.article-body', '.article-content',

            // 주요 언론사별 셀렉터
            '.news_view', '.article_txt', '.article_view', '.view_txt',
            '.read_body', '.article_area', '.news_article', '.content_area',
            '#content', '.content', '.post_content', '.entry_content',
            '.news_body', '.article_wrap', '.view_area', '.detail_content',

            // 모바일/반응형 셀렉터
            '.m_article', '.mobile_content', '.responsive_content',

            // 범용 셀렉터 (최후 수단)
            'article', 'main', '.main', '[class*="content"]', '[class*="article"]'
        ];

        // 먼저 네이버 뉴스 통합뷰에서 제목, 언론사, 날짜, 본문 모두 추출
        for (const selector of naverNewsSelectors) {
            const hasContent = await detailPage.$(selector);
            if (hasContent) {
                const content = await detailPage.$eval(selector, el => el.innerText.trim());
                if (content && content.length > 50) {
                    console.log(`네이버 통합뷰에서 본문 추출 성공 (${selector})`);

                    // 제목 추출
                    let title = '';
                    try {
                        title = await detailPage.$eval('.media_end_head_headline', el => el.textContent.trim());
                    } catch (e) {
                        // 제목을 찾을 수 없는 경우 다른 셀렉터 시도
                        try {
                            title = await detailPage.$eval('h1, .title, .headline', el => el.textContent.trim());
                        } catch (e2) {
                            title = '제목 없음';
                        }
                    }

                    // 언론사 추출
                    let press = '';
                    try {
                        const pressImg = await detailPage.$('.media_end_head_top_logo_img');
                        if (pressImg) {
                            press = await detailPage.$eval('.media_end_head_top_logo_img', el => el.alt || '');
                        }
                        if (!press) {
                            press = await detailPage.$eval('.media_end_head_top_logo, .press_name, .source', el => el.textContent.trim());
                        }
                    } catch (e) {
                        press = '언론사 없음';
                    }

                    // 날짜 추출
                    let dateText = '';
                    try {
                        dateText = await detailPage.$eval('._ARTICLE_DATE_TIME', el => el.textContent.trim());
                    } catch (e) {
                        try {
                            dateText = await detailPage.$eval('._ARTICLE_MODIFY_DATE_TIME', el => el.textContent.trim());
                        } catch (e2) {
                            try {
                                dateText = await detailPage.$eval('.media_end_head_info_datestamp_time', el => el.textContent.trim());
                            } catch (e3) {
                                dateText = '';
                            }
                        }
                    }

                    const cleanContent = content.replace(/\s+/g, ' ')
                        .replace(/\[.*?\]/g, '')
                        .replace(/\(.*?\)/g, '')
                        .replace(/\<.*?\>/g, '')
                        .replace(/본문 내용.*?보기/g, '')
                        .replace(/▶.*?$/gm, '')
                        .replace(/기자\s*=/g, '')
                        .replace(/사진.*?기자=/g, '')
                        .replace(/Copyright.*?Reserved\./gi, '')
                        .replace(/무단.*?금지/g, '')
                        .trim();

                    return {
                        title: title || '제목 없음',
                        press: press || '언론사 없음',
                        dateText: dateText || '',
                        content: cleanContent
                    };
                }
            }
        }

        // 네이버 통합뷰에서 못 찾은 경우 언론사 사이트 셀렉터로 시도
        console.log('네이버 통합뷰에서 본문을 찾지 못함. 언론사 사이트 시도.');
        for (const selector of pressSelectors) {
            try {
                const hasContent = await detailPage.$(selector);
                if (hasContent) {
                    const content = await detailPage.$eval(selector, el => {
                        // innerText와 textContent 모두 시도
                        let text = el.innerText || el.textContent || '';

                        // 불필요한 요소들 제거
                        const elementsToRemove = el.querySelectorAll('script, style, .ad, .advertisement, .related, .share, .comment, nav, header, footer');
                        elementsToRemove.forEach(elem => elem.remove());

                        // 다시 텍스트 추출
                        text = el.innerText || el.textContent || text;
                        return text.trim();
                    });

                    if (content && content.length > 100) { // 최소 길이를 100자로 증가
                        console.log(`언론사 사이트에서 본문 추출 성공 (${selector}) - ${content.length}자`);
                        const cleanContent = content
                            .replace(/\s+/g, ' ')
                            .replace(/\[.*?\]/g, '')
                            .replace(/\<.*?\>/g, '')
                            .replace(/본문 내용.*?보기/g, '')
                            .replace(/▶.*?$/gm, '')
                            .replace(/기자\s*[=:]/g, '')
                            .replace(/사진.*?기자[=:]/g, '')
                            .replace(/Copyright.*?Reserved\./gi, '')
                            .replace(/무단.*?금지/g, '')
                            .replace(/저작권자.*?무단.*?금지/g, '')
                            .replace(/\s*(더보기|관련기사|이전기사|다음기사)\s*/g, '')
                            .trim();
                        return {
                            title: '제목 추출 실패',
                            press: '언론사 추출 실패',
                            dateText: '',
                            content: cleanContent
                        };
                    }
                }
            } catch (selectorError) {
                // 개별 셀렉터 오류는 무시하고 다음 셀렉터 시도
                continue;
            }
        }
        return {
            title: '제목 추출 실패',
            press: '언론사 추출 실패',
            dateText: '',
            content: '본문을 가져올 수 없습니다.'
        };
    } catch (error) {
        console.log(`상세 내용 크롤링 실패: ${url} - ${error.message}`);
        return {
            title: '제목 추출 실패',
            press: '언론사 추출 실패',
            dateText: '',
            content: '본문을 가져올 수 없습니다.'
        };
    } finally {
        if (detailPage) {
            await detailPage.close();
        }
    }
}

async function crawlAndSave(stockName = "엔비디아", stockSymbol = "NVIDIA") {
    let browser;
    let client;

    try {
        // MongoDB 연결
        client = new MongoClient(uri);
        await client.connect();
        const db = client.db("newsDB");
        const collection = db.collection("news");

        console.log(`${stockName} 뉴스 크롤링 시작...`);

        // 브라우저 실행
        browser = await puppeteer.launch({
            headless: true,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu'
            ]
        });

        const page = await browser.newPage();
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');

        const BATCH_SIZE = 20;
        let totalProcessed = 0;
        let totalSaved = 0;
        let newArticlesCount = 0;
        const batchArticles = [];

        // 네이버 금융 뉴스 검색으로 변경
        const baseUrl = `https://finance.naver.com/news/news_search.naver`;
        console.log(`네이버 금융 뉴스 크롤링으로 변경`);

        const pagesToCrawl = 3; // 각 종목당 3페이지

        for (let pageNum = 1; pageNum <= pagesToCrawl; pageNum++) {
            // 직접 URL 구성 (인코딩 문제 해결)
            const today = new Date().toISOString().split('T')[0];
            
            // 날짜 범위 설정 (최근 1개월만 크롤링)
            const oneMonthAgo = new Date();
            oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
            const startDate = oneMonthAgo.toISOString().split('T')[0];
            
            // 한글 종목명을 EUC-KR로 인코딩
            const encodedStockName = encodeToEUCKR(stockName);
            const searchUrl = `${baseUrl}?rcdate=&q=${encodedStockName}&sm=all.basic&pd=1&stDateStart=${startDate}&stDateEnd=${today}&page=${pageNum}`;
            console.log(`\n페이지 ${pageNum} 크롤링 (${startDate} ~ ${today}): ${searchUrl}`);

            await new Promise(resolve => setTimeout(resolve, Math.random() * 3000 + 2000));
            const responseFromFetch = await fetchWithRetry(page, searchUrl);

            // 페이지 로딩 대기
            console.log("페이지 로딩 대기 중...");
            await new Promise(resolve => setTimeout(resolve, 2000));

            const htmlContent = await page.content();
            const $ = cheerio.load(htmlContent);

            let newsElements = [];

            // 네이버 금융 뉴스 전용 셀렉터
            const selectorGroups = [
                // 네이버 금융 뉴스 기사 제목 링크
                ['.articleSubject a'],
                // 백업 셀렉터
                ['.articleSubject', '.title', '.subject']
            ];

            for (const selectors of selectorGroups) {
                for (const selector of selectors) {
                    try {
                        newsElements = await page.$$(selector);
                        if (newsElements.length > 0) {
                            console.log(`[DEBUG] Found ${newsElements.length} news elements using selector: ${selector}`);
                            break;
                        }
                    } catch (selectorError) {
                        console.log(`셀렉터 오류 (${selector}): ${selectorError.message}, 다음 셀렉터 시도...`);
                        continue;
                    }
                }
                if (newsElements.length > 0) break;
            }

            if (newsElements.length === 0) {
                console.log("X 뉴스를 찾을 수 없습니다. HTML 저장...");
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const htmlContentForDebug = await page.content();
                fs.writeFileSync(`naver_debug_nodata_${timestamp}.html`, htmlContentForDebug);
                console.log(`HTML 저장됨: naver_debug_nodata_${timestamp}.html`);

                const pageTitle = await page.title();
                console.log(`페이지 제목: ${pageTitle}`);

                if (pageTitle.includes('차단') || pageTitle.includes('접근') || pageTitle.includes('block') || (responseFromFetch && responseFromFetch.status() === 403)) {
                    console.log(' --- 네이버 접근 차단 가능성 있음. 대기 필요 ---');
                    await new Promise(resolve => setTimeout(resolve, 60000));
                    return;
                }
            } // 이 부분이 원래 코드에서 닫는 괄호가 누락되어 있었습니다

            // 뉴스 요소들 처리
            for (let i = 0; i < newsElements.length; i++) {
                const el = newsElements[i];

                const articleData = await page.evaluate((element) => {
                    let title = '';
                    let link = '';
                    let summary = '';
                    let dateText = '';
                    let press = '';

                    // 네이버 금융 뉴스 전용 title과 link 추출
                    // element 자체가 이미 <a> 태그인 경우
                    if (element.tagName === 'A') {
                        title = element.textContent?.trim() || element.innerText?.trim() || '';
                        link = element.getAttribute('href') || '';
                    } else {
                        // element 내부에서 <a> 태그 찾기
                        const titleLink = element.querySelector('a');
                        if (titleLink) {
                            title = titleLink.textContent?.trim() || titleLink.innerText?.trim() || '';
                            link = titleLink.getAttribute('href') || '';
                        }
                    }

                    // 상대 URL을 절대 URL로 변환
                    if (link && !link.startsWith('http')) {
                        if (link.startsWith('/')) {
                            link = `https://finance.naver.com${link}`;
                        } else {
                            link = `https://finance.naver.com/${link}`;
                        }
                    }

                    // 네이버 금융 뉴스는 간단한 구조이므로 기본값만 설정
                    summary = ''; // 금융 뉴스에는 요약이 별도로 없음
                    press = '네이버 금융'; // 모든 기사가 네이버 금융에서 가져온 것
                    dateText = 'N/A'; // 일단 기본값, 나중에 개별 페이지에서 추출

                    return { title, link, summary, dateText, press };
                }, el); // el을 page.evaluate로 전달

                // 디버그: 추출된 정보 확인
                console.log(`\n[DEBUG] 기사 ${i + 1}:`);
                console.log(`  제목: "${articleData.title}"`);
                console.log(`  링크: "${articleData.link}"`);
                console.log(`  유효성: 제목=${!!articleData.title}, 링크=${!!articleData.link}, HTTP=${articleData.link?.startsWith('http')}`);

                if (!articleData.title || !articleData.link || !articleData.link.startsWith('http')) {
                    console.log(`  유효하지 않은 제목/링크가 있어서 스킵`);
                    continue;
                }

                const isDup = await isDuplicate(collection, articleData.title, articleData.link);
                if (isDup) {
                    console.log(`  중복 뉴스 스킵: "${articleData.title}" | ${articleData.link}`);
                    continue;
                }

                let fullContent = articleData.summary || "내용 없음";
                let actualTitle = articleData.title;
                let actualPress = articleData.press;
                let actualDate = articleData.dateText;

                if (articleData.link && articleData.link.includes('news.naver.com')) {
                    try {
                        const detailInfo = await getDetailContent(browser, articleData.link);
                        if (detailInfo.content !== "본문을 가져올 수 없습니다.") {
                            fullContent = detailInfo.content;
                            actualTitle = detailInfo.title || articleData.title;
                            actualPress = detailInfo.press || articleData.press;
                            actualDate = detailInfo.dateText || articleData.dateText;
                        }
                    } catch (detailError) {
                        console.log(`  상세 내용 크롤링 실패: ${detailError.message}`);
                    }
                    await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
                } else {
                    console.log(`  네이버 뉴스 링크가 아니어서 상세 내용 크롤링 스킵: ${articleData.link}`);
                }

                const article = {
                    stock: stockName, // 한글 종목명 사용
                    title: actualTitle,
                    content: fullContent,
                    summary: articleData.summary,
                    url: articleData.link,
                    press: actualPress,
                    published_at: parseKoreanDate(actualDate),
                    created_at: new Date()
                };

                batchArticles.push(article);

                // 최종 추출된 정보만 출력
                console.log(`\n[기사 ${i + 1}] ${actualTitle}`);
                console.log(`  언론사: ${actualPress} | 날짜: ${actualDate}`);
                console.log(`  수집 완료`);
                newArticlesCount++;
                totalProcessed++;

                if (batchArticles.length >= BATCH_SIZE) {
                    console.log(`\n[DEBUG] ${batchArticles.length}개 기사 배치 저장 중...`);
                    try {
                        const result = await collection.insertMany(batchArticles, { ordered: false });
                        const savedCount = result.insertedCount || 0;
                        totalSaved += savedCount;
                        console.log(`${savedCount}개 뉴스 저장 완료 (총 저장: ${totalSaved}개)`);
                        batchArticles.length = 0;
                        const count = await collection.countDocuments({ stock: stockName });
                        console.log(`[DEBUG] ${stockName} 총 문서 수: ${count}개`);
                    } catch (dbError) {
                        console.error('X MongoDB 저장 오류:', dbError.message);
                        if (dbError.writeErrors) {
                            console.error(`쓰기 오류 ${dbError.writeErrors.length}개 발생`);
                            dbError.writeErrors.slice(0, 3).forEach((err, idx) => {
                                console.error(`  오류 ${idx + 1}: ${err.errmsg}`);
                            });
                        }
                    }
                }
            }

            console.log(`페이지 ${pageNum} 완료 - 새로 수집: ${newArticlesCount}개`);

            // 페이지별 배치 저장 (20개마다)
            if (batchArticles.length >= BATCH_SIZE) {
                console.log(`\n[DEBUG] ${batchArticles.length}개 기사 배치 저장 중...`);
                try {
                    const result = await collection.insertMany(batchArticles, { ordered: false });
                    const savedCount = result.insertedCount || 0;
                    totalSaved += savedCount;
                    console.log(`${savedCount}개 뉴스 저장 완료 (총 저장: ${totalSaved}개)`);
                    batchArticles.length = 0;
                    const count = await collection.countDocuments({ stock: stockName });
                    console.log(`[DEBUG] ${stockName} 총 문서 수: ${count}개`);
                } catch (dbError) {
                    console.error('X MongoDB 저장 오류:', dbError.message);
                }
            }
        } // 페이지 루프 끝

        // 남은 배치 처리
        if (batchArticles.length > 0) {
            console.log(`\n[DEBUG] 남은 ${batchArticles.length}개 기사 배치 저장 중...`);
            try {
                const result = await collection.insertMany(batchArticles, { ordered: false });
                const savedCount = result.insertedCount || 0;
                totalSaved += savedCount;
                console.log(`${savedCount}개 뉴스 저장 완료 (총 저장: ${totalSaved}개)`);
                batchArticles.length = 0;
                const count = await collection.countDocuments({ stock: stockSymbol });
                console.log(`[DEBUG] ${stockSymbol} 총 문서 수: ${count}개`);
            } catch (dbError) {
                console.error('X MongoDB 마지막 배치 저장 오류:', dbError.message);
                if (dbError.writeErrors) {
                    console.error(`쓰기 오류 ${dbError.writeErrors.length}개 발생`);
                    dbError.writeErrors.slice(0, 3).forEach((err, idx) => {
                        console.error(`  오류 ${idx + 1}: ${err.errmsg}`);
                    });
                }
            }
        }

        console.log(`\n크롤링 완료 - 새로 수집: ${newArticlesCount}개`);
        console.log(`\n크롤링 완료!`);
        console.log(`- 처리된 총 기사: ${totalProcessed}개`);
        console.log(`- 저장된 새 기사: ${totalSaved}개`);

        if (totalSaved === 0) {
            console.log("X 저장할 새로운 뉴스가 없습니다. (모두 중복이거나 찾지 못함)");
            try {
                const existingCount = await collection.countDocuments({ stock: stockName });
                if (existingCount > 0) {
                    console.log(`💡 ${stockName} 관련 기사가 이미 ${existingCount}개 저장되어 있습니다.`);

                    const recentArticles = await collection.find({ stock: stockName })
                        .sort({ created_at: -1 })
                        .limit(3)
                        .toArray();

                    console.log('\n최근 저장된 기사들:');
                    recentArticles.forEach((article, idx) => {
                        console.log(`${idx + 1}. ${article.title}`);
                        console.log(`   날짜: ${article.published_at.toLocaleString()}`);
                        console.log(`   URL: ${article.url}`);
                    });
                } else {
                    console.log("💡 컬렉션이 비어있습니다. 네이버 접근이 차단되었거나, 뉴스 요소를 찾지 못했습니다.");
                }
            } catch (checkError) {
                console.error("컬렉션 확인 중 오류:", checkError.message);
            }
        }

    } catch (error) {
        console.error("X 크롤링 중 치명적인 오류 발생:", error.message);
        if (error.message.includes('ERR_TOO_MANY_REDIRECTS')) {
            console.error("너무 많은 리다이렉션 오류: 사이트가 크롤링을 차단하거나 잘못 설정되었을 수 있습니다.");
        } else if (error.message.includes('net::ERR_ABORTED')) {
            console.error("요청 중단 오류: 네트워크 문제 또는 브라우저에 의해 요청이 중단되었습니다.");
        } else if (error.message.includes('Timeout')) {
            console.error("페이지 로딩 타임아웃: 네트워크가 느리거나 사이트 로딩이 오래 걸립니다. 타임아웃 설정을 늘리거나 네트워크를 확인하세요.");
        }
        console.error("상세 오류:", error);
    } finally {
        // 리소스 정리
        try {
            if (browser) {
                console.log("브라우저 종료 중...");
                await browser.close();
                console.log("브라우저 종료 완료");
            }
        } catch (browserError) {
            console.error("브라우저 종료 중 오류:", browserError.message);
        }

        try {
            if (client) {
                console.log("MongoDB 연결 종료 중...");
                await client.close();
                console.log("MongoDB 연결이 종료되었습니다.");
            }
        } catch (mongoError) {
            console.error("MongoDB 연결 종료 중 오류:", mongoError.message);
        }
    }
}

// S&P 500 전체 크롤링 함수
async function crawlAllSP500() {
    try {
        // JSON 파일에서 종목 리스트 로드
        const stocksData = JSON.parse(fs.readFileSync('sp500_korean_stocks_clean.json', 'utf8'));
        const stocks = stocksData.stocks;

        const startFromStock = ""; // 시작할 종목명 (비워두면 처음부터)
        const startIndex = startFromStock ? stocks.findIndex(stock => stock.includes(startFromStock)) : 0;

        console.log(`S&P 500 전체 종목 크롤링 시작: ${stocks.length}개 종목`);
        console.log('각 종목당 3페이지씩 크롤링 진행...\n');

        let totalArticles = 0;

        for (let i = startIndex; i < stocks.length; i++) {
            const stockName = stocks[i];
            const stockIndex = i + 1;

            console.log(`\n[${'='.repeat(50)}]`);
            console.log(`[${stockIndex}/${stocks.length}] ${stockName} 크롤링 시작...`);
            console.log(`[${'='.repeat(50)}]`);

            try {
                await crawlAndSave(stockName, stockName);
                console.log(`✅ ${stockName} 크롤링 완료`);

                // 종목 간 대기 (네이버 차단 방지)
                const delay = Math.random() * 5000 + 3000; // 3-8초 랜덤 대기
                console.log(`⏳ 다음 종목까지 ${Math.round(delay/1000)}초 대기...`);
                await new Promise(resolve => setTimeout(resolve, delay));

            } catch (error) {
                console.error(`❌ ${stockName} 크롤링 실패: ${error.message}`);
                // 실패해도 다음 종목 계속 진행
                continue;
            }
        }

        console.log(`\n🎉 S&P 500 전체 크롤링 완료!`);
        console.log(`총 ${stocks.length}개 종목 처리됨`);

    } catch (error) {
        console.error('전체 크롤링 중 오류:', error.message);
    }
}

// 실행 부분
if (require.main === module) {
    // 개별 종목 크롤링 (월간 수집용)
    const targetStock = process.argv[2];
    
    if (targetStock) {
        console.log(`개별 종목 크롤링: ${targetStock}`);
        crawlAndSave(targetStock, targetStock);
    } else {
        // 기본값: 엔비디아만 크롤링 (테스트용)
        console.log("기본 크롤링: 엔비디아");
        crawlAndSave("엔비디아", "NVIDIA");
    }
    
    // 전체 S&P 500 크롤링은 주석 처리 (너무 많은 요청으로 차단될 수 있음)
    // crawlAllSP500();
}