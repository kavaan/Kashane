using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using HousePricePredictionAPI.Models;
using Swashbuckle.AspNetCore.Annotations;
using HousePricePredictionAPI.Models.HousePricePredictionAPI.Models;

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

            // مسیر فایل در wwwroot/Data
            string modelPath = Path.Combine(env.WebRootPath, "Data", "updated_house_data_babol.csv");

            // Load data and train the model
            IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(modelPath, separatorChar: ',', hasHeader: true);

            var pipeline = mlContext.Transforms.CopyColumns("Label", "Price")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Elevator"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Parking"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PropertyType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("DocumentType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("NearCenters"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RenovationStatus"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("FlooringType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("FacadeType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("ExtraAmenities"))
                .Append(mlContext.Transforms.Concatenate("Features", new[]
                {
                    "Neighborhood", "Area", "Rooms", "YearBuilt", "Floors", "FloorNumber",
                    "Elevator", "Parking", "PropertyType", "DocumentType", "NearCenters",
                    "RenovationStatus", "FlooringType", "FacadeType", "ExtraAmenities"
                }))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(dataView);

            _predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);
        }

        [HttpPost]
        [SwaggerOperation(
            Summary = "Predict the price of a house",
            Description = "Provide details about the house to predict its price."
        )]
        public ActionResult<float> Predict([FromBody] HouseInputData houseInput)
        {
            // تبدیل HouseInputData به HouseData بدون Price
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

            // قالب‌بندی قیمت به صورت سه رقم سه رقم و افزودن کلمه "تومان"
            string formattedPrice = $"{prediction.Price:N0} تومان";

            return Ok(new { PredictedPrice = formattedPrice });
        }
    }
}
