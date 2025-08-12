// ==================================================================================
// 매월 자동 뉴스 데이터 수집 스크립트
// 매월 1일에 실행하여 최신 뉴스를 MongoDB에 수집
// ==================================================================================

const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// 현재 날짜 기반으로 수집 범위 설정
function getDateRange() {
    const now = new Date();
    const currentMonth = now.getMonth();
    const currentYear = now.getFullYear();
    
    // 이번 달 1일부터 현재까지
    const startDate = new Date(currentYear, currentMonth, 1);
    const endDate = new Date(now);
    
    return {
        start: startDate.toISOString().split('T')[0],
        end: endDate.toISOString().split('T')[0]
    };
}

// 크롤링 실행
async function runMonthlyCrawling() {
    console.log('🚀 매월 뉴스 데이터 수집 시작...');
    console.log(`실행 날짜: ${new Date().toLocaleString()}`);
    
    const dateRange = getDateRange();
    console.log(`수집 범위: ${dateRange.start} ~ ${dateRange.end}`);
    
    try {
        // 1단계: 크롤링 스크립트 실행
        console.log('1단계: 뉴스 크롤링 중...');
        const crawlScript = path.join(__dirname, '../../crawling/crawl.js');
        
        await new Promise((resolve, reject) => {
            exec(`node "${crawlScript}"`, {
                cwd: path.join(__dirname, '../../crawling'),
                maxBuffer: 1024 * 1024 * 10 // 10MB buffer
            }, (error, stdout, stderr) => {
                if (error) {
                    console.error('크롤링 실행 실패:', error);
                    reject(error);
                    return;
                }
                
                console.log('크롤링 결과:');
                console.log(stdout);
                
                if (stderr) {
                    console.warn('크롤링 경고:', stderr);
                }
                
                console.log('크롤링 완료!');
                resolve();
            });
        });
        
        // 2단계: 뉴스 전처리 파이프라인 실행
        console.log('2단계: 뉴스 전처리 파이프라인 실행 중...');
        const processingScript = path.join(__dirname, './news_processing_pipeline.py');
        
        await new Promise((resolve, reject) => {
            exec(`python "${processingScript}"`, {
                cwd: __dirname,
                maxBuffer: 1024 * 1024 * 20 // 20MB buffer
            }, (error, stdout, stderr) => {
                if (error) {
                    console.error('전처리 파이프라인 실행 실패:', error);
                    reject(error);
                    return;
                }
                
                console.log('전처리 결과:');
                console.log(stdout);
                
                if (stderr) {
                    console.warn('전처리 경고:', stderr);
                }
                
                console.log('전처리 완료!');
                resolve();
            });
        });
        
        console.log('전체 데이터 수집 및 전처리 완료!');
        
    } catch (error) {
        console.error('데이터 수집 프로세스 실패:', error);
        throw error;
    }
}

// 로그 기록
function logExecution(success, error = null) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        success: success,
        error: error ? error.message : null
    };
    
    const logFile = path.join(__dirname, './logs/monthly_crawling.log');
    
    // logs 디렉토리 생성
    const logDir = path.dirname(logFile);
    if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
    }
    
    // 로그 추가
    fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');
}

// 메인 실행
async function main() {
    try {
        await runMonthlyCrawling();
        logExecution(true);
        console.log('매월 뉴스 수집 성공적 완료!');
    } catch (error) {
        console.error('매월 뉴스 수집 실패:', error.message);
        logExecution(false, error);
        process.exit(1);
    }
}

// 스크립트가 직접 실행될 때만 main 함수 호출
if (require.main === module) {
    main();
}