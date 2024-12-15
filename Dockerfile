# مرحله Build
FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src

COPY ["Kashane/Kashane.csproj", "./"]
RUN dotnet restore

COPY . .
RUN dotnet publish -c Release -o /app

# مرحله Runtime
FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS final
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "Kashane.dll"]
