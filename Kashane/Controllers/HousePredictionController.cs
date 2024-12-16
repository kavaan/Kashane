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
        public ActionResult<float> Predict([FromBody] HouseInputData houseInput)
        {
            var adjustedPrice = AdjustPrice(houseInput);
            return Ok(new { PredictedPrice = $"{adjustedPrice:N0} تومان" });
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

            // تاثیر آسانسور
            if (houseInput.Elevator == "Yes")
            {
                if (houseInput.Floors > 4)
                    finalPrice *= 1.10f; // افزایش 10 درصدی برای ساختمان‌های بلند
                else
                    finalPrice *= 1.05f; // افزایش 5 درصدی برای ساختمان‌های کوتاه
            }

            // تاثیر پارکینگ
            if (houseInput.Parking == "Yes")
            {
                if (houseInput.Area > 150)
                    finalPrice *= 1.07f; // افزایش 7 درصدی برای خانه‌های بزرگ
                else
                    finalPrice *= 1.12f; // افزایش 12 درصدی برای خانه‌های کوچک‌تر
            }

            // تاثیر مکان‌های نزدیک
            switch (houseInput.NearCenters)
            {
                case "School":
                    finalPrice *= 0.98f; // کاهش 2 درصدی برای نزدیکی به مدرسه
                    break;
                case "MainStreet":
                    finalPrice *= 1.20f; // افزایش 20 درصدی برای نزدیکی به خیابان اصلی
                    break;
                case "ShoppingCenter":
                    finalPrice *= 1.15f; // افزایش 15 درصدی برای نزدیکی به مرکز خرید
                    break;
                case "Park":
                    finalPrice *= 1.10f; // افزایش 10 درصدی برای نزدیکی به پارک
                    break;
            }

            // تاثیر نوع نما
            switch (houseInput.FacadeType)
            {
                case "Brick":
                    finalPrice *= 0.95f; // کاهش 5 درصدی برای نمای آجر
                    break;
                case "Cement":
                    finalPrice *= 0.97f; // کاهش 3 درصدی برای نمای سیمان
                    break;
                case "Stone":
                    finalPrice *= 1.12f; // افزایش 12 درصدی برای نمای سنگ
                    break;
            }

            // تاثیر نوع سند
            switch (houseInput.DocumentType)
            {
                case "SingleDeed":
                    finalPrice *= 1.08f; // افزایش 8 درصدی برای سند تک‌برگ
                    break;
                case "Agreement":
                    finalPrice *= 0.90f; // کاهش 10 درصدی برای قولنامه‌ای
                    break;
                case "FullOwnership":
                    finalPrice *= 1.15f; // افزایش 15 درصدی برای سند شش‌دانگ
                    break;
            }

            // تاثیر وضعیت بازسازی
            switch (houseInput.RenovationStatus)
            {
                case "New":
                    finalPrice *= 1.25f; // افزایش 25 درصدی برای خانه‌های نوساز
                    break;
                case "Renovated":
                    finalPrice *= 1.10f; // افزایش 10 درصدی برای خانه‌های بازسازی‌شده
                    break;
                case "Old":
                    finalPrice *= 0.80f; // کاهش 20 درصدی برای خانه‌های قدیمی
                    break;
            }

            // رابطه بین سال ساخت و وضعیت بازسازی
            int currentYear = DateTime.Now.Year;
            int age = currentYear - (int)houseInput.YearBuilt;

            if (age > 30 && houseInput.RenovationStatus == "Renovated")
            {
                finalPrice *= 1.05f; // افزایش 5 درصدی برای خانه‌های قدیمی که بازسازی شده‌اند
            }
            else if (age > 30 && houseInput.RenovationStatus == "Old")
            {
                finalPrice *= 0.70f; // کاهش 30 درصدی برای خانه‌های قدیمی بدون بازسازی
            }

            // تاثیر نوع کف‌پوش
            switch (houseInput.FlooringType)
            {
                case "Ceramic":
                    finalPrice *= 1.03f; // افزایش 3 درصدی برای کف‌پوش سرامیک
                    break;
                case "Parquet":
                    finalPrice *= 1.12f; // افزایش 12 درصدی برای کف‌پوش پارکت
                    break;
                case "Stone":
                    finalPrice *= 1.10f; // افزایش 10 درصدی برای کف‌پوش سنگ
                    break;
                case "Carpet":
                    finalPrice *= 0.90f; // کاهش 10 درصدی برای کف‌پوش موکت
                    break;
            }

            // تاثیر طبقه و رابطه آن با تعداد طبقات
            if (houseInput.Elevator == "No")
            {
                if (houseInput.FloorNumber == 2)
                    finalPrice *= 0.93f; // کاهش 7 درصدی برای طبقه دوم
                else if (houseInput.FloorNumber == 3)
                    finalPrice *= 0.88f; // کاهش 12 درصدی برای طبقه سوم
                else if (houseInput.FloorNumber >= 4)
                    finalPrice *= 0.80f; // کاهش 20 درصدی برای طبقه چهارم به بالا
            }
            else if (houseInput.FloorNumber >= 4)
            {
                finalPrice *= 1.07f; // افزایش 7 درصدی برای طبقه چهارم به بالا در ساختمان‌های دارای آسانسور
            }

            return finalPrice;
        }
    }
}
