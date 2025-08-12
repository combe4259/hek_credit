const puppeteer = require("puppeteer");
const fs = require('fs');

async function crawlSP500Stocks() {
    let browser;

    try {
        console.log('S&P 500 한글 종목명 + 심볼 크롤링 시작...');

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

        // 테이블 행(tr) 단위로 접근하여 한글 종목명 + 심볼 추출
        const tableRows = await page.$$('table tr');
        console.log(`${tableRows.length}개 테이블 행 발견`);

        // 각 행에서 한글 종목명과 심볼 추출
        for (let i = 0; i < tableRows.length; i++) {
            const row = tableRows[i];

            const stockInfo = await page.evaluate((rowElement) => {
                const cells = rowElement.querySelectorAll('td');
                let koreanName = null;
                let symbol = null;

                // 각 셀을 확인하여 한글 종목명과 심볼 찾기
                for (let j = 0; j < cells.length; j++) {
                    const cell = cells[j];
                    const cellText = cell.textContent?.trim() || '';

                    // 1. 한글 종목명 찾기
                    const hasKorean = /[가-힣]/.test(cellText);
                    if (hasKorean && cellText.length >= 2 && cellText.length <= 50 && !koreanName) {
                        // 한글 종목명만 추출 (티커 심볼과 줄바꿈 모두 제거)
                        let koreanOnly = cellText
                            .split('\n')[0] // 첫 번째 줄만 (한글 종목명)
                            .replace(/[A-Z0-9.\-]{2,}/g, '') // 티커 심볼 제거
                            .replace(/\s+/g, ' ') // 여러 공백을 하나로
                            .trim();

                        // 한글이 남아있고 적절한 길이인지 재확인
                        if (/[가-힣]/.test(koreanOnly) && koreanOnly.length >= 2 && koreanOnly.length <= 30) {
                            koreanName = koreanOnly;
                        }
                    }

                    // 2. 심볼 찾기 (class="symbol txt-en" 또는 일반적인 패턴)
                    const symbolElement = cell.querySelector('.symbol.txt-en, .symbol, .txt-en');
                    if (symbolElement && !symbol) {
                        const symbolText = symbolElement.textContent?.trim();
                        // 3-5자 대문자 알파벳으로 구성된 심볼
                        if (symbolText && /^[A-Z]{1,5}$/.test(symbolText)) {
                            symbol = symbolText;
                        }
                    }

                    // 3. 심볼이 없으면 셀 텍스트에서 직접 찾기
                    if (!symbol) {
                        // 3-5자 대문자 심볼 패턴 매칭
                        const symbolMatch = cellText.match(/\b([A-Z]{1,5})\b/);
                        if (symbolMatch && symbolMatch[1] && symbolMatch[1].length >= 1 && symbolMatch[1].length <= 5) {
                            // 일반적인 단어가 아닌 심볼인지 확인
                            const commonWords = ['TD', 'CEO', 'USA', 'NYSE', 'USD'];
                            if (!commonWords.includes(symbolMatch[1])) {
                                symbol = symbolMatch[1];
                            }
                        }
                    }
                }

                // 한글명과 심볼이 모두 있으면 반환
                if (koreanName && symbol) {
                    return {
                        korean_name: koreanName,
                        symbol: symbol
                    };
                }

                return null;
            }, row);

            if (stockInfo && !stocks.find(s => s.symbol === stockInfo.symbol)) {
                stocks.push(stockInfo);
                console.log(`${stocks.length}. ${stockInfo.korean_name} (${stockInfo.symbol})`);

                // 600개 정도에서 멈춤 (안전장치)
                if (stocks.length >= 600) {
                    console.log('600개 도달, 크롤링 중단');
                    break;
                }
            }
        }

        // 심볼만 있고 한글명이 없는 경우를 위한 추가 처리
        console.log('\n🔍 심볼만 있는 항목 추가 검색...');

        const additionalSymbols = await page.evaluate(() => {
            const symbolElements = document.querySelectorAll('.symbol.txt-en, .symbol, .txt-en');
            const foundSymbols = [];

            symbolElements.forEach(element => {
                const symbolText = element.textContent?.trim();
                if (symbolText && /^[A-Z]{1,5}$/.test(symbolText)) {
                    foundSymbols.push(symbolText);
                }
            });

            return [...new Set(foundSymbols)]; // 중복 제거
        });

        // 기존에 없는 심볼들 추가
        additionalSymbols.forEach(symbol => {
            if (!stocks.find(s => s.symbol === symbol)) {
                stocks.push({
                    korean_name: `Unknown-${symbol}`, // 한글명을 찾지 못한 경우
                    symbol: symbol
                });
                console.log(`${stocks.length}. Unknown-${symbol} (${symbol}) - 심볼만 발견`);
            }
        });

        // 결과 저장
        if (stocks.length > 0) {
            const stocksData = {
                crawled_at: new Date().toISOString(),
                total_count: stocks.length,
                stocks: stocks.map(stock => ({
                    korean_name: stock.korean_name,
                    symbol: stock.symbol,
                    has_korean_name: !stock.korean_name.startsWith('Unknown-')
                }))
            };

            // JSON 파일 저장
            fs.writeFileSync('sp500_korean_stocks_with_symbols.json', JSON.stringify(stocksData, null, 2), 'utf8');

            // CSV 파일도 저장 (엑셀에서 보기 편함)
            const csvHeader = 'Korean Name,Symbol,Has Korean Name\n';
            const csvData = stocksData.stocks.map(stock =>
                `"${stock.korean_name}","${stock.symbol}","${stock.has_korean_name}"`
            ).join('\n');
            fs.writeFileSync('sp500_korean_stocks_with_symbols.csv', csvHeader + csvData, 'utf8');

            console.log(`\n✅ 크롤링 완료: ${stocks.length}개 종목 저장`);
            console.log('저장 파일:');
            console.log('  - sp500_korean_stocks_with_symbols.json');
            console.log('  - sp500_korean_stocks_with_symbols.csv');

            // 통계 출력
            const withKoreanName = stocks.filter(s => !s.korean_name.startsWith('Unknown-')).length;
            const symbolOnly = stocks.length - withKoreanName;

            console.log(`\n📊 수집 통계:`);
            console.log(`  - 한글명 + 심볼: ${withKoreanName}개`);
            console.log(`  - 심볼만: ${symbolOnly}개`);

            // 샘플 출력
            console.log('\n📋 수집된 종목 샘플 (처음 10개):');
            stocks.slice(0, 10).forEach((stock, idx) => {
                console.log(`${idx + 1}. ${stock.korean_name} (${stock.symbol})`);
            });

            return stocks;
        } else {
            console.log('❌ 종목 정보를 찾지 못했습니다.');
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

// 특정 심볼들을 수동으로 추가하는 함수 (필요시)
function addManualSymbols() {
    const manualSymbols = [
        { korean_name: '애플', symbol: 'AAPL' },
        { korean_name: '마이크로소프트', symbol: 'MSFT' },
        { korean_name: '엔비디아', symbol: 'NVDA' },
        { korean_name: '구글(알파벳)', symbol: 'GOOGL' },
        { korean_name: '테슬라', symbol: 'TSLA' },
        { korean_name: '메타(페이스북)', symbol: 'META' },
        { korean_name: '아마존', symbol: 'AMZN' }
        // 필요한 종목들 추가...
    ];

    try {
        const existingData = JSON.parse(fs.readFileSync('sp500_korean_stocks_with_symbols.json', 'utf8'));

        manualSymbols.forEach(manual => {
            const existingIndex = existingData.stocks.findIndex(s => s.symbol === manual.symbol);
            if (existingIndex >= 0) {
                // 기존 항목 업데이트
                existingData.stocks[existingIndex].korean_name = manual.korean_name;
                existingData.stocks[existingIndex].has_korean_name = true;
                console.log(`업데이트: ${manual.korean_name} (${manual.symbol})`);
            } else {
                // 새 항목 추가
                existingData.stocks.push({
                    korean_name: manual.korean_name,
                    symbol: manual.symbol,
                    has_korean_name: true
                });
                console.log(`추가: ${manual.korean_name} (${manual.symbol})`);
            }
        });

        existingData.total_count = existingData.stocks.length;
        fs.writeFileSync('sp500_korean_stocks_with_symbols.json', JSON.stringify(existingData, null, 2), 'utf8');
        console.log('✅ 수동 심볼 추가 완료');

    } catch (error) {
        console.error('수동 심볼 추가 오류:', error.message);
    }
}

// 실행
if (require.main === module) {
    crawlSP500Stocks().then(() => {
        console.log('\n🔧 수동 심볼 추가 실행...');
        addManualSymbols();
    });
}

module.exports = { crawlSP500Stocks, addManualSymbols };