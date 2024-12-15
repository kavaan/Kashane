FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src

# Copy the project files
COPY Kashane/Kashane.csproj ./

# Restore dependencies
RUN dotnet restore

# Copy the entire source code
COPY Kashane/ ./

# Publish the application
RUN dotnet publish -c Release -o /app

# Use the ASP.NET runtime image
FROM mcr.microsoft.com/dotnet/aspnet:7.0
WORKDIR /app
COPY --from=build /app .

# Expose port 80
EXPOSE 80

# Set the entry point
ENTRYPOINT ["dotnet", "Kashane.dll"]
