const { MongoClient } = require("mongodb");

// MongoDB connection
const uri = "mongodb+srv://julk0206:%23Sooyeon2004@hek.yqi7d9x.mongodb.net";
const client = new MongoClient(uri);

async function checkFullContent() {
    try {
        await client.connect();
        const db = client.db("newsDB");
        const collection = db.collection("news");

        // 최근 5개 기사의 전체 본문 확인
        console.log("=== 최근 수집된 기사의 전체 본문 확인 ===\n");
        const recentArticles = await collection.find({ stock: "NVIDIA" })
            .sort({ created_at: -1 })
            .limit(5)
            .toArray();

        recentArticles.forEach((article, index) => {
            console.log(`${index + 1}. ${article.title}`);
            console.log(`URL: ${article.url}`);
            console.log(`본문 길이: ${article.content ? article.content.length : 0}자`);
            console.log("전체 본문:");
            console.log("=" + "=".repeat(80));
            console.log(article.content || "본문 없음");
            console.log("=" + "=".repeat(80));
            console.log("\n");
        });

        // 본문 길이별 통계
        console.log("=== 본문 길이별 통계 ===");
        const contentStats = await collection.aggregate([
            { $match: { stock: "NVIDIA" } },
            {
                $project: {
                    contentLength: {
                        $cond: {
                            if: { $eq: ["$content", null] },
                            then: 0,
                            else: { $strLenCP: "$content" }
                        }
                    }
                }
            },
            {
                $group: {
                    _id: {
                        $switch: {
                            branches: [
                                { case: { $eq: ["$contentLength", 0] }, then: "본문 없음" },
                                { case: { $lt: ["$contentLength", 100] }, then: "100자 미만" },
                                { case: { $lt: ["$contentLength", 300] }, then: "100-300자" },
                                { case: { $lt: ["$contentLength", 500] }, then: "300-500자" },
                                { case: { $lt: ["$contentLength", 1000] }, then: "500-1000자" },
                                { case: { $gte: ["$contentLength", 1000] }, then: "1000자 이상" }
                            ],
                            default: "기타"
                        }
                    },
                    count: { $sum: 1 },
                    avgLength: { $avg: "$contentLength" }
                }
            },
            { $sort: { count: -1 } }
        ]).toArray();

        contentStats.forEach(stat => {
            console.log(`${stat._id}: ${stat.count}개 (평균 ${Math.round(stat.avgLength)}자)`);
        });

        // 본문이 너무 짧은 기사들 확인
        console.log("\n=== 본문이 짧은 기사들 (200자 미만) ===");
        const shortArticles = await collection.find({
            stock: "NVIDIA",
            $expr: { $lt: [{ $strLenCP: "$content" }, 200] }
        }).limit(5).toArray();

        shortArticles.forEach((article, index) => {
            console.log(`${index + 1}. ${article.title} (${article.content ? article.content.length : 0}자)`);
            console.log(`   ${article.content}`);
            console.log("");
        });

    } catch (error) {
        console.error("본문 확인 중 오류:", error);
    } finally {
        await client.close();
    }
}

checkFullContent();