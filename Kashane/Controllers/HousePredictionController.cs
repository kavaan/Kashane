using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using HousePricePredictionAPI.Models;
using Swashbuckle.AspNetCore.Annotations;
using System;
using System.IO;
using HousePricePredictionAPI.Models.HousePricePredictionAPI.Models;
using Microsoft.ML.Trainers.FastTree;


namespace HousePricePredictionAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class HousePredictionController : ControllerBase
    {
        private readonly PredictionEngine<HouseData, HousePricePrediction> _predictionEngine;

        public HousePredictionController(IWebHostEnvironment env)
        {
            var mlContext = new MLContext();
            var model = TrainModel(mlContext, env);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);
        }
        private string GenerateAnalysis(HouseInputData houseInput, float predictedPrice)
        {
            var analysis = "تحلیل قیمت خانه:\n";

            analysis += $"- محله: {houseInput.Neighborhood}\n";
            analysis += $"- متراژ: {houseInput.Area} متر مربع\n";
            analysis += $"- تعداد اتاق: {houseInput.Rooms}\n";
            analysis += $"- سال ساخت: {houseInput.YearBuilt}\n";

            if (houseInput.Elevator == "Yes")
            {
                analysis += "- این خانه دارای آسانسور است که باعث افزایش قیمت می‌شود.\n";
            }
            else
            {
                analysis += "- عدم وجود آسانسور ممکن است قیمت را کاهش دهد، مخصوصاً برای طبقات بالا.\n";
            }

            if (houseInput.Parking == "Yes")
            {
                analysis += "- وجود پارکینگ یک مزیت مهم برای این خانه محسوب می‌شود.\n";
            }

            if (houseInput.NearCenters == "MainStreet")
            {
                analysis += "- نزدیکی به خیابان اصلی باعث افزایش قابل توجه قیمت می‌شود.\n";
            }

            analysis += $"\nقیمت پیش‌بینی شده نهایی: {predictedPrice:N0} تومان";

            return analysis;
        }

        private ITransformer TrainModel(MLContext mlContext, IWebHostEnvironment env)
        {
            string modelPath = Path.Combine(env.WebRootPath, "Data", "updated_house_data_babol.csv");

            // بارگذاری داده‌ها
            IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(modelPath, separatorChar: ',', hasHeader: true);

            // پیش‌پردازش داده‌ها
            var pipeline = mlContext.Transforms.CopyColumns("Label", "Price")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PropertyType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("DocumentType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("NearCenters"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RenovationStatus"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("FlooringType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("FacadeType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("ExtraAmenities"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Elevator"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Parking"))
                .Append(mlContext.Transforms.Concatenate("Features", new[]
                {
                    "Neighborhood", "Area", "Rooms", "YearBuilt", "Floors", "FloorNumber",
                    "Elevator", "Parking", "PropertyType", "DocumentType", "NearCenters",
                    "RenovationStatus", "FlooringType", "FacadeType", "ExtraAmenities"
                }))
                .Append(mlContext.Regression.Trainers.FastTree());

            // آموزش مدل
            var model = pipeline.Fit(dataView);
            return model;
        }

        [HttpPost]
        [SwaggerOperation(
            Summary = "Predict the price of a house",
            Description = "Provide details about the house to predict its price."
        )]
        [HttpPost]
        public ActionResult Predict([FromBody] HouseInputData houseInput)
        {
            var adjustedPrice = AdjustPrice(houseInput);
            var analysis = GenerateAnalysis(houseInput, adjustedPrice);

            return Ok(new
            {
                PredictedPrice = $"{adjustedPrice:N0} تومان",
                Analysis = analysis
            });
        }

        private float AdjustPrice(HouseInputData houseInput)
        {
            var houseData = new HouseData
            {
                Neighborhood = houseInput.Neighborhood,
                Area = houseInput.Area,
                Rooms = houseInput.Rooms,
                YearBuilt = houseInput.YearBuilt,
                Floors = houseInput.Floors,
                FloorNumber = houseInput.FloorNumber,
                Elevator = houseInput.Elevator,
                Parking = houseInput.Parking,
                PropertyType = houseInput.PropertyType,
                DocumentType = houseInput.DocumentType,
                NearCenters = houseInput.NearCenters,
                RenovationStatus = houseInput.RenovationStatus,
                FlooringType = houseInput.FlooringType,
                FacadeType = houseInput.FacadeType,
                ExtraAmenities = houseInput.ExtraAmenities
            };

            var prediction = _predictionEngine.Predict(houseData);
            float finalPrice = prediction.Price;

            // تاثیر منفی طبقه در صورت نبود آسانسور
            if (houseInput.Elevator == "No")
            {
                if (houseInput.FloorNumber == 2)
                    finalPrice *= 0.96f; // کاهش 4 درصدی برای طبقه دوم
                else if (houseInput.FloorNumber == 3)
                    finalPrice *= 0.92f; // کاهش 8 درصدی برای طبقه سوم
                else if (houseInput.FloorNumber >= 4)
                    finalPrice *= 0.88f; // کاهش 12 درصدی برای طبقه چهارم به بالا
            }

            // تاثیر مثبت آسانسور
            if (houseInput.Elevator == "Yes")
            {
                finalPrice *= 1.045f; // افزایش 4.5 درصدی به دلیل وجود آسانسور
            }

            // تاثیر پارکینگ
            if (houseInput.Parking == "Yes")
            {
                if (houseInput.Area > 150)
                    finalPrice *= 1.045f; // افزایش 4.5 درصدی برای خانه‌های بزرگ
                else
                    finalPrice *= 1.07f; // افزایش 7 درصدی برای خانه‌های کوچک‌تر
            }

            // تاثیر مکان‌های نزدیک
            switch (houseInput.NearCenters)
            {
                case "School":
                case "Hospital":
                    finalPrice *= 0.99f; // کاهش 1 درصدی
                    break;
                case "MainStreet":
                    finalPrice *= 1.09f; // افزایش 9 درصدی
                    break;
                case "ShoppingCenter":
                    finalPrice *= 1.07f; // افزایش 7 درصدی
                    break;
                case "Park":
                    finalPrice *= 1.045f; // افزایش 4.5 درصدی
                    break;
            }

            // تاثیر نوع نما
            switch (houseInput.FacadeType)
            {
                case "Brick":
                    finalPrice *= 0.98f; // کاهش 2 درصدی
                    break;
                case "Cement":
                    finalPrice *= 0.985f; // کاهش 1.5 درصدی
                    break;
                case "Stone":
                    finalPrice *= 1.055f; // افزایش 5.5 درصدی
                    break;
            }

            // تاثیر نوع سند
            switch (houseInput.DocumentType)
            {
                case "SingleDeed":
                    finalPrice *= 1.035f; // افزایش 3.5 درصدی
                    break;
                case "Agreement":
                    finalPrice *= 0.96f; // کاهش 4 درصدی
                    break;
                case "FullOwnership":
                    finalPrice *= 1.07f; // افزایش 7 درصدی
                    break;
            }

            // تاثیر وضعیت بازسازی
            switch (houseInput.RenovationStatus)
            {
                case "New":
                    finalPrice *= 1.13f; // افزایش 13 درصدی
                    break;
                case "Renovated":
                    finalPrice *= 1.045f; // افزایش 4.5 درصدی
                    break;
                case "Old":
                    finalPrice *= 0.88f; // کاهش 12 درصدی
                    break;
            }

            // رابطه بین سال ساخت و وضعیت بازسازی
            int currentYear = DateTime.Now.Year;
            int age = currentYear - (int)houseInput.YearBuilt;

            if (age > 30 && houseInput.RenovationStatus == "Renovated")
            {
                finalPrice *= 1.025f; // افزایش 2.5 درصدی
            }
            else if (age > 30 && houseInput.RenovationStatus == "Old")
            {
                finalPrice *= 0.80f; // کاهش 20 درصدی
            }

            // تاثیر نوع کف‌پوش
            switch (houseInput.FlooringType)
            {
                case "Ceramic":
                    finalPrice *= 1.018f; // افزایش 1.8 درصدی
                    break;
                case "Parquet":
                    finalPrice *= 1.05f; // افزایش 5 درصدی
                    break;
                case "Stone":
                    finalPrice *= 1.045f; // افزایش 4.5 درصدی
                    break;
                case "Carpet":
                    finalPrice *= 0.96f; // کاهش 4 درصدی
                    break;
            }

            return finalPrice;
        }
    }
}
