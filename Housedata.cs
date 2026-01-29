using Microsoft.ML.Data;

public class HouseData
{
    [LoadColumn(0)]
    public float Area;

    [LoadColumn(1)]
    public float Rooms;

    [LoadColumn(2)]
    public float Age;

    [LoadColumn(3)]
    public float Price;
}

public class HousePrediction
{
    [ColumnName("Score")]
    public float Price;
}

