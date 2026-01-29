// See https://aka.ms/new-console-template for more information
using System;
using Microsoft.ML;

class Program
{
    static void Main(string[] args)
    {
        // 1. Create ML Context
        var mlContext = new MLContext();

        // 2. Load Data
        var data = mlContext.Data.LoadFromTextFile<HouseData>(
            "house_data.csv",     
            separatorChar: ',',
            hasHeader: true);

        // 3. Build Pipeline
        var pipeline = mlContext.Transforms.Concatenate(
                     "Features",
                     nameof(HouseData.Area),
                     nameof(HouseData.Rooms),
                     nameof(HouseData.Age))
                 .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                 .Append(mlContext.Regression.Trainers.FastTree(
                     labelColumnName: nameof(HouseData.Price)));


        // 4. Train model
        var model = pipeline.Fit(data);
        Console.WriteLine("Model trained successfully!");

        // 5. Save model
        mlContext.Model.Save(model, data.Schema, "houseModel.zip");
        Console.WriteLine("Model saved as houseModel.zip!");

        // 6. Load model for prediction
        var loadedModel = mlContext.Model.Load("houseModel.zip", out _);
        var engine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(loadedModel);

        // 7. Get input from user
        Console.Write("Enter Area: ");
        float area = float.Parse(Console.ReadLine());

        Console.Write("Enter Rooms: ");
        float rooms = float.Parse(Console.ReadLine());

        Console.Write("Enter Age: ");
        float age = float.Parse(Console.ReadLine());

        var input = new HouseData()
        {
            Area = area,
            Rooms = rooms,
            Age = age
        };

        // 8. Predict
        var prediction = engine.Predict(input);
        Console.WriteLine($"Predicted House Price: {prediction.Price}");
        Console.WriteLine("Press any key to exit...");
        Console.ReadLine();
    }
}

