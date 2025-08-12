const { MongoClient } = require("mongodb");

// MongoDB connection
const uri = "mongodb+srv://julk0206:%23Sooyeon2004@hek.yqi7d9x.mongodb.net";
const client = new MongoClient(uri);

async function checkData() {
    try {
        await client.connect();
        const db = client.db("newsDB");
        const collection = db.collection("news");

        // 총 개수 확인
        const totalCount = await collection.countDocuments({ stock: "NVIDIA" });
        console.log(`총 NVIDIA 관련 기사 수: ${totalCount}개`);

        // 최근 10개 기사 확인
        console.log("\n=== 최근 수집된 기사 10개 ===");
        const recentArticles = await collection.find({ stock: "NVIDIA" })
            .sort({ created_at: -1 })
            .limit(10)
            .toArray();

        recentArticles.forEach((article, index) => {
            console.log(`\n${index + 1}. ${article.title}`);
            console.log(`   언론사: ${article.press}`);
            console.log(`   발행일: ${article.published_at.toLocaleString()}`);
            console.log(`   URL: ${article.url}`);
            console.log(`   본문 길이: ${article.content ? article.content.length : 0}자`);
            if (article.content && article.content.length > 0) {
                console.log(`   본문 미리보기: ${article.content.substring(0, 100)}...`);
            }
        });

        // 언론사별 통계
        console.log("\n=== 언론사별 기사 수 ===");
        const pressCounts = await collection.aggregate([
            { $match: { stock: "NVIDIA" } },
            { $group: { _id: "$press", count: { $sum: 1 } } },
            { $sort: { count: -1 } },
            { $limit: 10 }
        ]).toArray();

        pressCounts.forEach(press => {
            console.log(`${press._id}: ${press.count}개`);
        });

        // 유효한 본문이 있는 기사 수
        const validContentCount = await collection.countDocuments({
            stock: "NVIDIA",
            $and: [
                { content: { $ne: "본문을 가져올 수 없습니다." } },
                { content: { $ne: "내용 없음" } },
                { content: { $ne: null } },
                { content: { $ne: "" } }
            ]
        });
        console.log(`\n유효한 본문이 있는 기사: ${validContentCount}개`);

    } catch (error) {
        console.error("데이터 확인 중 오류:", error);
    } finally {
        await client.close();
    }
}

checkData();