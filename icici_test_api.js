var axios = require('axios');

var config = {
    method: 'get',
    url: 'https://breezeapi.icicidirect.com/api/v2/historicalcharts?stock_code=NIFTY&exch_code=NFO&from_date=2022-11-10 09:15:00&to_date=2022-11-11 09:16:00&interval=day&product_type=Options&expiry_date=2022-11-24&right=Call&strike_price=18000',
    headers: { 
        'X-SessionToken': '', 
        'apikey': ''
    }
};

axios(config)
.then(function (response) {
    console.log(JSON.stringify(response.data));
})
.catch(function (error) {
    console.log(error);
});
