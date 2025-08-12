const puppeteer = require("puppeteer");
const fs = require('fs');

async function crawlSP500Stocks() {
    let browser;
    
    try {
        console.log('S&P 500 한글 종목명 크롤링 시작...');
        
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
        
        const url = 'https://www.hankyung.com/globalmarket/usa-stock-sp500';
        console.log(`페이지 로딩: ${url}`);
        
        await page.goto(url, {
            waitUntil: 'networkidle2',
            timeout: 60000
        });

        // 페이지 로딩 대기
        await new Promise(resolve => setTimeout(resolve, 3000));

        const stocks = [];
        
        // 테이블 행(tr) 단위로 접근하여 한글 종목명만 추출
        const tableRows = await page.$$('table tr');
        console.log(`${tableRows.length}개 테이블 행 발견`);

        // 각 행에서 한글 종목명 추출
        for (let i = 0; i < tableRows.length; i++) {
            const row = tableRows[i];
            
            const stockName = await page.evaluate((rowElement) => {
                const cells = rowElement.querySelectorAll('td');
                
                // 각 셀을 확인하여 한글 종목명 찾기
                for (let j = 0; j < cells.length; j++) {
                    const cellText = cells[j].textContent?.trim() || '';
                    
                    // 한글이 포함된 텍스트 찾기
                    const hasKorean = /[가-힣]/.test(cellText);
                    
                    if (hasKorean && cellText.length >= 2 && cellText.length <= 50) {
                        // 한글 종목명만 추출 (티커 심볼과 줄바꿈 모두 제거)
                        let koreanOnly = cellText
                            .split('\n')[0] // 첫 번째 줄만 (한글 종목명)
                            .replace(/[A-Z0-9.\-]{2,}/g, '') // 티커 심볼 제거
                            .replace(/\s+/g, ' ') // 여러 공백을 하나로
                            .trim();
                        
                        // 한글이 남아있고 적절한 길이인지 재확인
                        if (/[가-힣]/.test(koreanOnly) && koreanOnly.length >= 2 && koreanOnly.length <= 30) {
                            return koreanOnly;
                        }
                    }
                }
                
                return null;
            }, row);

            if (stockName && !stocks.includes(stockName)) {
                stocks.push(stockName);
                console.log(`${stocks.length}. ${stockName}`);
                
                // 600개 정도에서 멈춤 (안전장치)
                if (stocks.length >= 600) {
                    console.log('600개 도달, 크롤링 중단');
                    break;
                }
            }
        }

        // 결과 저장
        if (stocks.length > 0) {
            const stocksData = {
                crawled_at: new Date().toISOString(),
                total_count: stocks.length,
                stocks: stocks
            };
            
            fs.writeFileSync('sp500_korean_stocks_clean.json', JSON.stringify(stocksData, null, 2), 'utf8');
            console.log(`\n✅ 크롤링 완료: ${stocks.length}개 한글 종목명 저장`);
            console.log('저장 파일: sp500_korean_stocks_clean.json');
            
            // 샘플 출력
            console.log('\n📋 수집된 종목 샘플 (처음 10개):');
            stocks.slice(0, 10).forEach((stock, idx) => {
                console.log(`${idx + 1}. ${stock}`);
            });
            
            return stocks;
        } else {
            console.log('❌ 한글 종목명을 찾지 못했습니다.');
            return [];
        }

    } catch (error) {
        console.error('크롤링 오류:', error.message);
        return [];
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

// 실행
if (require.main === module) {
    crawlSP500Stocks();
}

module.exports = { crawlSP500Stocks };