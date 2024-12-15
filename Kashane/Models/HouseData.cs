using Microsoft.ML.Data;

namespace HousePricePredictionAPI.Models
{
    using global::HousePricePredictionAPI.Models.HousePricePredictionAPI.Models;
    using Microsoft.ML.Data;

    namespace HousePricePredictionAPI.Models
    {
        public class HouseInputData
        {
            [LoadColumn(0)]
            public string Neighborhood { get; set; }

            [LoadColumn(1)]
            public float Area { get; set; }

            [LoadColumn(2)]
            public float Rooms { get; set; }

            [LoadColumn(3)]
            public float YearBuilt { get; set; }

            [LoadColumn(4)]
            public float Floors { get; set; }

            [LoadColumn(5)]
            public float FloorNumber { get; set; }

            [LoadColumn(6)]
            public string Elevator { get; set; }

            [LoadColumn(7)]
            public string Parking { get; set; }

            [LoadColumn(8)]
            public string PropertyType { get; set; }

            [LoadColumn(9)]
            public string DocumentType { get; set; }

            [LoadColumn(10)]
            public string NearCenters { get; set; }

            [LoadColumn(11)]
            public string RenovationStatus { get; set; }

            [LoadColumn(12)]
            public string FlooringType { get; set; }

            [LoadColumn(13)]
            public string FacadeType { get; set; }

            [LoadColumn(14)]
            public string ExtraAmenities { get; set; }
        }
    }
    public class HouseData : HouseInputData
    {
        [LoadColumn(15)]
        public float Price { get; set; } // قیمت
    }
    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
