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

            // تاثیر منفی طبقه در صورت نبود آسانسور
            if (houseInput.Elevator == "No")
            {
                if (houseInput.FloorNumber == 2)
                    finalPrice *= 0.93f; // کاهش 7 درصدی برای طبقه دوم
                else if (houseInput.FloorNumber == 3)
                    finalPrice *= 0.88f; // کاهش 12 درصدی برای طبقه سوم
                else if (houseInput.FloorNumber >= 4)
                    finalPrice *= 0.80f; // کاهش 20 درصدی برای طبقه چهارم به بالا
            }

            // تاثیر مثبت آسانسور به عنوان یک مزیت مستقل
            if (houseInput.Elevator == "Yes")
            {
                finalPrice *= 1.10f; // افزایش 10 درصدی به دلیل وجود آسانسور
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
                case "Hospital":
                    finalPrice *= 0.98f;
                    break;
                case "MainStreet":
                    finalPrice *= 1.20f;
                    break;
                case "ShoppingCenter":
                    finalPrice *= 1.15f;
                    break;
                case "Park":
                    finalPrice *= 1.10f;
                    break;
            }

            // تاثیر نوع نما
            switch (houseInput.FacadeType)
            {
                case "Brick":
                    finalPrice *= 0.95f;
                    break;
                case "Cement":
                    finalPrice *= 0.97f;
                    break;
                case "Stone":
                    finalPrice *= 1.12f;
                    break;
            }

            // تاثیر نوع سند
            switch (houseInput.DocumentType)
            {
                case "SingleDeed":
                    finalPrice *= 1.08f;
                    break;
                case "Agreement":
                    finalPrice *= 0.90f;
                    break;
                case "FullOwnership":
                    finalPrice *= 1.15f;
                    break;
            }

            // تاثیر وضعیت بازسازی
            switch (houseInput.RenovationStatus)
            {
                case "New":
                    finalPrice *= 1.25f;
                    break;
                case "Renovated":
                    finalPrice *= 1.10f;
                    break;
                case "Old":
                    finalPrice *= 0.80f;
                    break;
            }

            // رابطه بین سال ساخت و وضعیت بازسازی
            int currentYear = DateTime.Now.Year;
            int age = currentYear - (int)houseInput.YearBuilt;

            if (age > 30 && houseInput.RenovationStatus == "Renovated")
            {
                finalPrice *= 1.05f;
            }
            else if (age > 30 && houseInput.RenovationStatus == "Old")
            {
                finalPrice *= 0.70f;
            }

            // تاثیر نوع کف‌پوش
            switch (houseInput.FlooringType)
            {
                case "Ceramic":
                    finalPrice *= 1.03f;
                    break;
                case "Parquet":
                    finalPrice *= 1.12f;
                    break;
                case "Stone":
                    finalPrice *= 1.10f;
                    break;
                case "Carpet":
                    finalPrice *= 0.90f;
                    break;
            }

            return finalPrice;
        }
    }
}
