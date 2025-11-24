
class WeatherRecord:
    def __init__(self, date="", city="", temperature=0.0):
        self.date = date        # format: DD/MM/YYYY
        self.city = city
        self.temperature = temperature



class WeatherDataStorage:
    def __init__(self, years, cities):
        self.years = years
        self.cities = cities
        self.SENTINEL = -9999   # for sparse data
        # 2D array: years x cities
        self.temperatureData = [
            [self.SENTINEL for _ in range(len(cities))]
            for _ in range(len(years))
        ]

    
    def insert(self, record: WeatherRecord):
        year = int(record.date[6:10])   # extract YYYY
        row = self.getYearIndex(year)
        col = self.getCityIndex(record.city)

        if row != -1 and col != -1:
            self.temperatureData[row][col] = record.temperature
        else:
            print("Invalid city or year!")

    
    def remove(self, city, year):
        row = self.getYearIndex(year)
        col = self.getCityIndex(city)

        if row != -1 and col != -1:
            self.temperatureData[row][col] = self.SENTINEL

    
    def retrieve(self, city, year):
        row = self.getYearIndex(year)
        col = self.getCityIndex(city)

        if row != -1 and col != -1:
            if self.temperatureData[row][col] != self.SENTINEL:
                print(
                    f"Temperature in {city} ({year}): {self.temperatureData[row][col]}°C"
                )
            else:
                print("No data available.")

    
    def populateArray(self):
        for i in range(len(self.years)):
            for j in range(len(self.cities)):
                self.temperatureData[i][j] = (i + j) * 2 + 20  # dummy values

    
    def rowMajorAccess(self):
        print("\nRow-Major Traversal:")
        for i in range(len(self.years)):
            for j in range(len(self.cities)):
                print(
                    f"[{self.years[i]},{self.cities[j]}]: {self.temperatureData[i][j]}\t",
                    end="",
                )
            print()

    
    def columnMajorAccess(self):
        print("\nColumn-Major Traversal:")
        for j in range(len(self.cities)):
            for i in range(len(self.years)):
                print(
                    f"[{self.years[i]},{self.cities[j]}]: {self.temperatureData[i][j]}\t",
                    end="",
                )
            print()

    
    def handleSparseData(self):
        print("\nSparse Data Representation:")
        for i in range(len(self.years)):
            for j in range(len(self.cities)):
                if self.temperatureData[i][j] != self.SENTINEL:
                    print(
                        f"[{self.years[i]},{self.cities[j]}]: {self.temperatureData[i][j]}"
                    )

    
    def analyzeComplexity(self):
        print("\nTime Complexity:")
        print("Insert: O(1)\nRetrieve: O(1)\nDelete: O(1)")
        print("\nSpace Complexity:")
        print("O(n*m) where n = years, m = cities")

    
    def getCityIndex(self, city):
        if city in self.cities:
            return self.cities.index(city)
        return -1

    def getYearIndex(self, year):
        if year in self.years:
            return self.years.index(year)
        return -1



if __name__ == "__main__":
    years = [2023, 2024, 2025]
    cities = ["Delhi", "Mumbai", "Chennai"]

    system = WeatherDataStorage(years, cities)

    
    system.populateArray()
    system.rowMajorAccess()
    system.columnMajorAccess()

    
    r1 = WeatherRecord("01/01/2024", "Delhi", 28.5)
    system.insert(r1)
    system.retrieve("Delhi", 2024)

    
    system.remove("Mumbai", 2025)
    system.handleSparseData()

    
    system.analyzeComplexity()

