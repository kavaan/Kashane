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

            // تنظیم قیمت بر اساس ویژگی‌های مختلف

            // تاثیر آسانسور
            if (houseInput.Elevator == "Yes")
                finalPrice *= 1.05f;

            // تاثیر پارکینگ
            if (houseInput.Parking == "Yes")
                finalPrice *= 1.10f;

            // تاثیر مکان‌های نزدیک
            switch (houseInput.NearCenters)
            {
                case "School":
                    finalPrice *= 0.95f;
                    break;
                case "MainStreet":
                    finalPrice *= 1.15f;
                    break;
                case "ShoppingCenter":
                case "Park":
                    finalPrice *= 1.10f;
                    break;
            }

            // تاثیر نوع نما
            switch (houseInput.FacadeType)
            {
                case "Brick":
                    finalPrice *= 0.90f;
                    break;
                case "Cement":
                    finalPrice *= 0.92f;
                    break;
                case "Stone":
                    finalPrice *= 1.08f;
                    break;
            }

            return finalPrice;
        }
    }
}
