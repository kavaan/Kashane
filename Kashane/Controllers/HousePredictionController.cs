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
                    finalPrice *= 0.958f; // کاهش 4.2 درصدی (30% کمتر از 6%)
                else if (houseInput.FloorNumber == 3)
                    finalPrice *= 0.923f; // کاهش 7.7 درصدی (30% کمتر از 11%)
                else if (houseInput.FloorNumber >= 4)
                    finalPrice *= 0.895f; // کاهش 10.5 درصدی (30% کمتر از 15%)
            }

            // تاثیر مثبت آسانسور
            if (houseInput.Elevator == "Yes")
            {
                finalPrice *= 1.028f; // افزایش 2.8 درصدی (30% کمتر از 4%)
            }

            // تاثیر پارکینگ
            if (houseInput.Parking == "Yes")
            {
                if (houseInput.Area > 150)
                    finalPrice *= 1.028f; // افزایش 2.8 درصدی (30% کمتر از 4%)
                else
                    finalPrice *= 1.042f; // افزایش 4.2 درصدی (30% کمتر از 6%)
            }

            // تاثیر مکان‌های نزدیک
            switch (houseInput.NearCenters)
            {
                case "School":
                case "Hospital":
                    finalPrice *= 0.986f; // کاهش 1.4 درصدی (30% کمتر از 2%)
                    break;
                case "MainStreet":
                    finalPrice *= 1.056f; // افزایش 5.6 درصدی (30% کمتر از 8%)
                    break;
                case "ShoppingCenter":
                    finalPrice *= 1.042f; // افزایش 4.2 درصدی (30% کمتر از 6%)
                    break;
                case "Park":
                    finalPrice *= 1.028f; // افزایش 2.8 درصدی (30% کمتر از 4%)
                    break;
            }

            // تاثیر نوع نما
            switch (houseInput.FacadeType)
            {
                case "Brick":
                    finalPrice *= 0.968f; // کاهش 3.2 درصدی (30% کمتر از 4%)
                    break;
                case "Cement":
                    finalPrice *= 0.979f; // کاهش 2.1 درصدی (30% کمتر از 3%)
                    break;
                case "Stone":
                    finalPrice *= 1.035f; // افزایش 3.5 درصدی (30% کمتر از 5%)
                    break;
            }

            // تاثیر نوع سند
            switch (houseInput.DocumentType)
            {
                case "SingleDeed":
                    finalPrice *= 1.021f; // افزایش 2.1 درصدی (30% کمتر از 3%)
                    break;
                case "Agreement":
                    finalPrice *= 0.958f; // کاهش 4.2 درصدی (30% کمتر از 6%)
                    break;
                case "FullOwnership":
                    finalPrice *= 1.042f; // افزایش 4.2 درصدی (30% کمتر از 6%)
                    break;
            }

            // تاثیر وضعیت بازسازی
            switch (houseInput.RenovationStatus)
            {
                case "New":
                    finalPrice *= 1.084f; // افزایش 8.4 درصدی (30% کمتر از 12%)
                    break;
                case "Renovated":
                    finalPrice *= 1.028f; // افزایش 2.8 درصدی (30% کمتر از 4%)
                    break;
                case "Old":
                    finalPrice *= 0.902f; // کاهش 9.8 درصدی (30% کمتر از 14%)
                    break;
            }

            // رابطه بین سال ساخت و وضعیت بازسازی
            int currentYear = DateTime.Now.Year;
            int age = currentYear - (int)houseInput.YearBuilt;

            if (age > 30 && houseInput.RenovationStatus == "Renovated")
            {
                finalPrice *= 1.014f; // افزایش 1.4 درصدی (30% کمتر از 2%)
            }
            else if (age > 30 && houseInput.RenovationStatus == "Old")
            {
                finalPrice *= 0.846f; // کاهش 15.4 درصدی (30% کمتر از 22%)
            }

            // تاثیر نوع کف‌پوش
            switch (houseInput.FlooringType)
            {
                case "Ceramic":
                    finalPrice *= 1.011f; // افزایش 1.1 درصدی (30% کمتر از 1.6%)
                    break;
                case "Parquet":
                    finalPrice *= 1.031f; // افزایش 3.1 درصدی (30% کمتر از 4.5%)
                    break;
                case "Stone":
                    finalPrice *= 1.028f; // افزایش 2.8 درصدی (30% کمتر از 4%)
                    break;
                case "Carpet":
                    finalPrice *= 0.958f; // کاهش 4.2 درصدی (30% کمتر از 6%)
                    break;
            }

            return finalPrice;
        }



    }
}
