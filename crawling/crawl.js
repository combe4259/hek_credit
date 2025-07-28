const puppeteer = require("puppeteer");
const cheerio = require("cheerio"); // Cheerio 모듈 추가
const { MongoClient } = require("mongodb");
const https = require('https'); // 이 모듈은 현재 직접 사용되지 않지만, 기존 코드에 있어 남겨둡니다.
const fs = require('fs');

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

// MongoDB connection
const uri = "mongodb+srv://julk0206:%23Sooyeon2004@hek.yqi7d9x.mongodb.net";
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
    return !!existing;
}

// 뉴스 디테일 본문 크롤링 (Puppeteer 기반)
async function getDetailContent(browser, url) {
    let detailPage;
    try {
        detailPage = await browser.newPage();
        await detailPage.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');
        
        await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));
        
        await detailPage.goto(url, {
            waitUntil: 'networkidle2',
            timeout: 60000
        });

        const contentSelectors = [
            '#dic_area', '._article_body_contents',
            '#articeBody', '.article_body', '.news_contents',
            '.end_contents_body', '#articleBodyContents', '.go_trans._article_content'
        ];

        for (const selector of contentSelectors) {
            const hasContent = await detailPage.$(selector);
            if (hasContent) {
                const content = await detailPage.$eval(selector, el => el.innerText.trim());
                if (content && content.length > 50) {
                    const cleanContent = content.replace(/\s+/g, ' ')
                                                .replace(/\[.*?\]/g, '')
                                                .replace(/\(.*?\)/g, '')
                                                .replace(/\<.*?\>/g, '')
                                                .replace(/본문 내용 크게 보기/g, '')
                                                .replace(/▶/g, '')
                                                .replace(/기자\s*=/g, '')
                                                .replace(/사진출처=(.*?)\s*기자=/g, '')
                                                .replace(/Copyright ⓒ .*? All Rights Reserved\./g, '')
                                                .trim();
                    return cleanContent;
                }
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

async function crawlAndSave(stockName = "엔비디아 NVIDIA", stockSymbol = "NVIDIA") {
    let browser;
    try {
        await client.connect();
        const db = client.db("newsDB");
        const collection = db.collection("news");

        console.log(`${stockName} 뉴스 크롤링 시작...`);

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
        const pagesToCrawl = 10;

        for (let pageNum = 1; pageNum <= pagesToCrawl; pageNum++) {
            let newArticlesCount = 0;
            const batchArticles = [];

            const start = (pageNum - 1) * 10 + 1;
            const searchUrl = `https://search.naver.com/search.naver?where=news&query=${encodeURIComponent(stockName)}&sort=1&photo=0&field=0&pd=0&ds=&de=&cluster_rank=74&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all&start=${start}`;

            await new Promise(resolve => setTimeout(resolve, Math.random() * 5000 + 3000));

            const responseFromFetch = await fetchWithRetry(page, searchUrl);

            // 첫 페이지 디버그 저장
            if (pageNum === 1) {
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const htmlContentForDebug = await page.content();
                fs.writeFileSync(`naver_debug_page1_${timestamp}.html`, htmlContentForDebug);
                console.log(`[DEBUG] 첫 페이지 HTML 저장됨: naver_debug_page1_${timestamp}.html`);
            }

            const htmlContent = await page.content();
            const $ = cheerio.load(htmlContent);

            let newsElements = [];
            const mainNewsListContainer = await page.$('ul.list_news._infinite_list'); // Outer container

            if (mainNewsListContainer) {
                // Try the most specific, observed new structure first based on user's HTML
                newsElements = await mainNewsListContainer.$$('div.sds-comps-vertical-layout._sghYQmdqcpm83O1jqen');
                if (newsElements.length === 0) {
                    newsElements = await mainNewsListContainer.$$('div[data-sds-comp="ArticleItem"]'); // Existing fallback
                }
                if (newsElements.length === 0) {
                    newsElements = await mainNewsListContainer.$$('div.sds-comps-base-layout.sds-comps-full-layout');
                }
                if (newsElements.length === 0) {
                    newsElements = await mainNewsListContainer.$$('li.sa_item_lazy_loading_wrap');
                }
                if (newsElements.length === 0) {
                    newsElements = await mainNewsListContainer.$$('li.bx');
                }
            } else {
                console.log("[DEBUG] 메인 뉴스 UL 컨테이너 없음. 대체 셀렉터 사용.");
                // Same priority for page-wide search if main container isn't found
                newsElements = await page.$$('div.sds-comps-vertical-layout._sghYQmdqcpm83O1jqen');
                if (newsElements.length === 0) {
                    newsElements = await page.$$('div[data-sds-comp="ArticleItem"]');
                }
                if (newsElements.length === 0) {
                    newsElements = await page.$$('div.sds-comps-base-layout.sds-comps-full-layout');
                }
                if (newsElements.length === 0) {
                    newsElements = await page.$$('li.sa_item_lazy_loading_wrap');
                }
                if (newsElements.length === 0) {
                    newsElements = await page.$$('li.bx');
                }
            }
            
            if (newsElements.length === 0) {
                console.log("[DEBUG] 일반 뉴스 셀렉터로 시도.");
                newsElements = await page.$$('.news_area .news_info_group, .news_area .news_result_item, .news_area .card_item, .news_wrap');
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
                    continue;
                }
                if (pageNum > 1) {
                    console.log("이전 페이지에서는 뉴스 있었지만 현재 페이지부터는 없음. 크롤링 종료.");
                    break;
                }
                continue;
            }

            for (let i = 0; i < newsElements.length; i++) {
                const el = newsElements[i];

                const articleData = await page.evaluate((element) => {
                    let title = '';
                    let link = '';
                    let summary = '';
                    let dateText = '';
                    let press = '';

                    // Title and Link
                    const newTitleLinkElement = element.querySelector('a[nocr="1"]');
                    if (newTitleLinkElement) {
                        const titleSpan = newTitleLinkElement.querySelector('span.sds-comps-text-type-headline1') ||
                                          newTitleLinkElement.querySelector('span.sds-comps-text-ellipsis');
                        if (titleSpan) title = titleSpan.textContent?.trim() || '';
                        link = newTitleLinkElement.getAttribute('href') || '';
                    } else {
                        const titleElement = element.querySelector('a.sa_text_title') ||
                                             element.querySelector('a.news_tit') ||
                                             element.querySelector('a[href*="news.naver.com"]');
                        if (titleElement) {
                            title = titleElement.textContent?.trim() || '';
                            link = titleElement.getAttribute('href') || '';
                        }
                    }
                    
                    // Summary
                    summary = element.querySelector('span.sds-comps-text.sds-comps-text-ellipsis.sds-comps-text-ellipsis-3')?.textContent?.trim() ||
                              element.querySelector('span.sds-comps-text-type-body1')?.textContent?.trim() ||
                              element.querySelector('.sa_text_desc')?.textContent?.trim() ||
                              element.querySelector('.api_txt_lines.dsc_txt_wrap')?.textContent?.trim() ||
                              element.querySelector('.dsc_txt_wrap')?.textContent?.trim() ||
                              element.querySelector('.news_dsc')?.textContent?.trim() ||
                              element.querySelector('.txt_inline')?.textContent?.trim() ||
                              element.querySelector('p.dsc_txt')?.textContent?.trim() || '';

                    // === 언론사/날짜 추출 로직 (최적화) ===
                    // 언론사 (Press)
                    let tempPress = '';
                    // 1. 가장 명확한 sds-comps-profile-info-title-text 찾기
                    const pressSpan1 = element.querySelector('span.sds-comps-profile-info-title-text');
                    if (pressSpan1) tempPress = pressSpan1.textContent?.trim();

                    // 2. div.sds-comps-profile-info-title 내의 sds-comps-text-type-body1
                    if (!tempPress) {
                        const pressSpan2 = element.querySelector('div.sds-comps-profile-info-title span.sds-comps-text-type-body1');
                        if (pressSpan2) tempPress = pressSpan2.textContent?.trim();
                    }
                    // 3. div.sds-comps-profile-source 내의 sds-comps-text-type-body1
                    if (!tempPress) {
                        const pressSpan3 = element.querySelector('div.sds-comps-profile-source span.sds-comps-text-type-body1');
                        if (pressSpan3) tempPress = pressSpan3.textContent?.trim();
                    }
                    // 4. sds-comps-profile-info-title div 자체의 텍스트
                    if (!tempPress) {
                        const pressDiv1 = element.querySelector('div.sds-comps-profile-info-title');
                        if (pressDiv1) tempPress = pressDiv1.textContent?.trim();
                    }
                    // 5. sds-comps-profile-source div 자체의 텍스트
                    if (!tempPress) {
                        const pressDiv2 = element.querySelector('div.sds-comps-profile-source');
                        if (pressDiv2) tempPress = pressDiv2.textContent?.trim();
                    }

                    // 6. 기존 폴백 셀렉터들
                    if (!tempPress) {
                        const pressOld1 = element.querySelector('.press');
                        if (pressOld1) tempPress = pressOld1.textContent?.trim();
                    }
                    if (!tempPress) {
                        const pressOld2 = element.querySelector('.info_group .press');
                        if (pressOld2) tempPress = pressOld2.textContent?.trim();
                    }
                    if (!tempPress) {
                        const pressOld3 = element.querySelector('span.sp_txt'); // 일반 정보 스팬
                        if (pressOld3) tempPress = pressOld3.textContent?.trim();
                    }
                    if (!tempPress) {
                        const pressOld4 = element.querySelector('.source');
                        if (pressOld4) tempPress = pressOld4.textContent?.trim();
                    }
                    press = tempPress || '';


                    // 날짜 (Date)
                    let tempDateText = '';
                    // 1. 가장 명확한 sds-comps-profile-info-subtext 찾기
                    const dateSpan1 = element.querySelector('span.sds-comps-profile-info-subtext');
                    if (dateSpan1) tempDateText = dateSpan1.textContent?.trim();

                    // 2. sds-comps-text-type-body2 중 날짜 패턴 포함하는 것 찾기
                    if (!tempDateText) {
                        const dateSpanCandidates = [...element.querySelectorAll('span.sds-comps-text-type-body2')];
                        const foundDateSpan = dateSpanCandidates.find(el => {
                            const text = el.textContent?.trim();
                            return text && text.match(/(\d{4}\.\d{2}\.\d{2}\.)|(\d+분 전|\d+시간 전|\d+일 전)/);
                        });
                        if (foundDateSpan) tempDateText = foundDateSpan.textContent?.trim();
                    }

                    // 3. 기존 폴백 셀렉터들
                    if (!tempDateText) {
                        const dateOld1 = element.querySelector('.info:last-child');
                        if (dateOld1) tempDateText = dateOld1.textContent?.trim();
                    }
                    if (!tempDateText) {
                        const dateOld2 = element.querySelector('.press_date');
                        if (dateOld2) tempDateText = dateOld2.textContent?.trim();
                    }
                    if (!tempDateText) {
                        const dateOld3 = element.querySelector('.date');
                        if (dateOld3) tempDateText = dateOld3.textContent?.trim();
                    }

                    // 4. 일반적인 span 요소 전체에서 날짜 패턴 찾기
                    if (!tempDateText) {
                        const allSpans = element.querySelectorAll('span');
                        for (const spanEl of allSpans) {
                            const text = spanEl.textContent?.trim();
                            if (text && text.match(/(\d{4}\.\d{2}\.\d{2}\.)|(\d+분 전|\d+시간 전|\d+일 전)/)) {
                                tempDateText = text;
                                break;
                            }
                        }
                    }
                    dateText = tempDateText || '';


                    // 최종 필터링 및 정리 (언론사)
                    // "아이뉴스24네이버뉴스"와 같이 붙어 나오는 경우, "네이버뉴스"를 제거
                    if (press && press.endsWith('네이버뉴스')) {
                        press = press.replace(/네이버뉴스$/, '').trim();
                    }
                    if (press && (press.length > 25 || title.includes(press) || summary.includes(press))) {
                        press = ""; // 유효하지 않은 언론사로 판단, 초기화
                    }
                    if (!press) {
                        // A 태그 중에서 링크가 기사 링크가 아니고, 텍스트가 짧고 날짜 패턴이 아닌 것을 언론사 후보로
                        const pressCandidateLink = [...element.querySelectorAll('a[href]')].find(aEl => {
                            const href = aEl.getAttribute('href');
                            const text = aEl.textContent?.trim();
                            return href && !href.includes('/article/') && !href.includes('news.naver.com/main/read') &&
                                   text && text.length > 1 && text.length < 20 &&
                                   !text.match(/(\d{4}\.\d{2}\.\d{2}\.)|(\d+분 전|\d+시간 전|\d+일 전)/) &&
                                   !title.includes(text) && !summary.includes(text);
                        });
                        if (pressCandidateLink) {
                            press = pressCandidateLink.textContent?.trim() || '';
                        }
                    }
                    if (!press) press = "알 수 없음"; // 최종적으로 못 찾으면 "알 수 없음"


                    // 최종 필터링 및 정리 (날짜)
                    // "네이버뉴스" 문자가 날짜에 포함되는 경우 제거
                    if (dateText && dateText.includes('네이버뉴스')) {
                        dateText = dateText.replace(/네이버뉴스/, '').trim();
                    }
                    if (dateText && (dateText.length > 20 || dateText.includes(press))) {
                        dateText = ""; // 유효하지 않은 날짜로 판단, 초기화
                    }
                    if (!dateText) dateText = 'N/A'; // 최종적으로 못 찾으면 "N/A"

                    return { title, link, summary, dateText, press };
                }, el); // el을 page.evaluate로 전달

                console.log(`\n[DEBUG] 기사 ${i + 1}:`);
                console.log(`  제목: ${articleData.title || 'N/A'}`);
                console.log(`  링크: ${articleData.link || 'N/A'}`);
                console.log(`  날짜: ${articleData.dateText || 'N/A'}`);
                console.log(`  언론사: ${articleData.press || 'N/A'}`);
                console.log(`  (현재 추출된 제목: ${articleData.title}, 링크: ${articleData.link})`);

                if (!articleData.title || !articleData.link || !articleData.link.startsWith('http')) {
                    console.log(`  ⏭️  유효하지 않은 제목/링크가 있어서 스킵`);
                    continue;
                }

                const isDup = await isDuplicate(collection, articleData.title, articleData.link);
                if (isDup) {
                    console.log(`  ⏭️  중복 뉴스 스킵`);
                    continue;
                }

                let fullContent = articleData.summary || "내용 없음";
                if (articleData.link && articleData.link.includes('news.naver.com')) {
                    console.log(`  상세 내용 크롤링 중...`);
                    try {
                        const detailContent = await getDetailContent(browser, articleData.link);
                        if (detailContent !== "본문을 가져올 수 없습니다.") {
                            fullContent = detailContent;
                        } else {
                            console.log(`  경고: 상세 내용 가져오기 실패, 요약 사용.`);
                        }
                    } catch (detailError) {
                        console.log(`  상세 내용 크롤링 실패 (getDetailContent 오류): ${detailError.message}`);
                    }
                    await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
                } else {
                    console.log(`  네이버 뉴스 링크가 아니어서 상세 내용 크롤링 스킵: ${articleData.link}`);
                }

                const article = {
                    stock: stockSymbol,
                    title: articleData.title,
                    content: fullContent,
                    summary: articleData.summary,
                    url: articleData.link,
                    press: articleData.press,
                    published_at: parseKoreanDate(articleData.dateText),
                    created_at: new Date()
                };

                batchArticles.push(article);
                console.log(`  수집 완료`);
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
                        const count = await collection.countDocuments({ stock: stockSymbol });
                        console.log(`[DEBUG] ${stockSymbol} 총 문서 수: ${count}개`);
                    } catch (dbError) {
                        console.error('X MongoDB 저장 오류:', dbError.message);
                        if (dbError.writeErrors) {
                            console.error(`쓰기 오류 ${dbError.writeErrors.length}개 발생`);
                            dbError.writeErrors.slice(0, 3).forEach((err, idx) => {
                                console.error(`  오류 ${idx + 1}: ${err.errmsg}`);
                            });
                        }
                    }
                }
            }

            if (batchArticles.length > 0) {
                console.log(`\n[DEBUG] 페이지 ${pageNum}의 남은 ${batchArticles.length}개 기사 배치 저장 중...`);
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
                            console.error(`  오류 ${idx + 1}: ${err.errmsg}`);
                        });
                    }
                }
            }

            console.log(`\n페이지 ${pageNum} 완료 - 새로 수집: ${newArticlesCount}개`);
        }

        console.log(`\n크롤링 완료!`);
        console.log(`- 처리된 총 기사: ${totalProcessed}개`);
        console.log(`- 저장된 새 기사: ${totalSaved}개`);

        if (totalSaved === 0) {
            console.log("X 저장할 새로운 뉴스가 없습니다. (모두 중복이거나 찾지 못함)");
            try {
                const existingCount = await collection.countDocuments({ stock: stockSymbol });
                if (existingCount > 0) {
                    console.log(`💡 ${stockSymbol} 관련 기사가 이미 ${existingCount}개 저장되어 있습니다.`);

                    const recentArticles = await collection.find({ stock: stockSymbol })
                        .sort({ created_at: -1 })
                        .limit(3)
                        .toArray();

                    console.log('\n최근 저장된 기사들:');
                    recentArticles.forEach((article, idx) => {
                        console.log(`${idx + 1}. ${article.title}`);
                        console.log(`   날짜: ${article.published_at.toLocaleString()}`);
                        console.log(`   URL: ${article.url}`);
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
        if (browser) {
            await browser.close();
        }
        await client.close();
        console.log("MongoDB 연결이 종료되었습니다.");
    }
}

// 여러 종목 크롤링
async function crawlMultipleStocks() {
    const stocks = [
        { name: "엔비디아", symbol: "NVIDIA" },
        { name: "테슬라", symbol: "TESLA" },
        { name: "애플", symbol: "APPLE" },
        { name: "삼성전자", symbol: "SAMSUNG" },
        { name: "SK하이닉스", symbol: "SKHYNIX" }
    ];

    for (const stock of stocks) {
        console.log(`\n${'='.repeat(50)}`);
        console.log(` ${stock.name} (${stock.symbol}) 크롤링 시작`);
        console.log('='.repeat(50));

        await crawlAndSave(stock.name, stock.symbol);

        await new Promise(resolve => setTimeout(resolve, 5000 + Math.random() * 5000));
    }
    console.log(`\n${'#'.repeat(50)}`);
    console.log(` 모든 종목 크롤링 완료`);
    console.log(`${'#'.repeat(50)}\n`);
}

if (require.main === module) {
    crawlAndSave("엔비디아 NVIDIA", "NVIDIA");
    // crawlMultipleStocks();
}