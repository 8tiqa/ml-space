// =============================================================================
/**
 * Cloud Computing Cource Exercises
 * Exercise 1
 *  2 Tasks
 *      1. Accessing VM information using unauthenticated API
 *      2. Service Level Authentication
 * Developed by 'Write Group Name'
 * Write Names of All Members
 */
// =============================================================================
/**
 * BASE SETUP
 * import the packages we need
 */
const express    = require('express');
const app        = express();
const port       = process.env.PORT || 8080; // set our port
/**
 * ROUTES FOR OUR API
 * Create our router
 */
const router = express.Router();
/**
 * Middleware to use for all requests
 */
router.use(function(req, res, next) {
    /**
     * Logs can be printed here while accessing any routes
     */
    console.log('Accessing Exercises Routes');
    next();
});
/**
 * Base route of the router : to make sure everything is working check http://localhost:8080/exercises)
 */
router.get('/', function(req, res) {
    res.json({ message: 'Welcome to Cloud Computing Exercises API!'});
});
/**
 * Exercise 1: Task 1 Route (Accessing VM information, This is also unauthenticated API)
 */

router.route('/exercise1_task1')
    .get(function(req, res)
    {
        /**
         * Hint : http://nodejs.org/api.html#_child_processes
         */
        var child_process = require("child_process");
        var numberUsers = parseInt(child_process.execSync("users | wc -w"));
        var userNames=((child_process.execSync("users").toString('utf-8')).replace(/\n/g,'')).split(' ');
        var numStorageDisks = parseInt(child_process.execSync("lsblk | grep disk | wc -l"));
        var storageDisksInfo=child_process.execSync("lsblk -o SIZE,TYPE | grep part").toString('utf-8').replace(/\n/g,'').split(" ");
        for(var i=0;i<(storageDisksInfo.length);i++)
        {
            if(storageDisksInfo[i].length==0)
            {
                storageDisksInfo.splice(i, 1);
            }
        }


        for(var i=0;i<(storageDisksInfo.length);i++)
        {
            if(storageDisksInfo[i].indexOf('part')>=0)
            {
                storageDisksInfo[i]=storageDisksInfo[i].replace('part','');
            }
        }
        for(var i=0;i<(storageDisksInfo.length);i++)
        {
            if(storageDisksInfo[i]==(''))
            {
                storageDisksInfo.splice(i,1)
            }
        }

// ================================================================================================================
        /**
         * TO DO
         * 1. Get the number of current users login into virtual machine
         * 2. Get the names of those users
         * 3. Get the number of storage disks ((we are here only concerned about the disks and that too Virtual disks (vd)))
         * 4. Get size Information about the above disks (disk: size).
         * 5. save in exercise_1_Message
         */
            // =================================================================================================================
        let exercise_1_Message = {
                message: 'exercise_1',
                numberUsers: numberUsers,
                userNames:userNames,
                numStorageDisks:numStorageDisks,
                storageDisksInfo:storageDisksInfo
            };
        res.json( exercise_1_Message);

    });
/**
 * Exercise 1: Task 2 Route (Service Level Authentication)
 */
router.route('/exercise1_task2')
    .get(function(req, res)
    {
        // ================================================================================================================
        /**
         * TO DO
         * 1. Add the default authentication to username: 'CCS' and password as 'CCS_exercise1_task2'.
         * 2. On success authentication return the response with value 'Successful Authentication'.
         * 3. In case of failure return the response with value 'Unsuccessful Authentication'.
         */
            // =================================================================================================================
        let auth;
        let allowed=false;
        /**
         * check whether an autorization header was send
         */
        if (req.headers.authorization)
        {
            /**
             *  only accepting basic auth, so:
             * cut the starting 'Basic ' from the header
             * decode the base64 encoded username:password
             * split the string at the colon
             * should result in an array
             */
            auth = new Buffer(req.headers.authorization.substring(6), 'base64').toString().split(':');
        }

        if(typeof auth != 'undefined')
        {
            if (auth[0]==='CCS'&&auth[1]==='CCS_exercise1_task2')
            {
                allowed=true;
            }
        }



        /**
         *  checks if:
         * auth array exists
         * first value matches the expected username
         * second value the expected password
         */
        if (allowed=false) {
            res.send('Unsuccessful Authentication');
        }
        else {
            /**
             * Processing can be continued here, user was authenticated
             */
            res.send('Successful Authentication');
        }
    });
/**
 * REGISTER OUR ROUTES
 * our router is now pointing to /exercises
 */
app.use('/exercises', router);
/**
 * Start the server
 * our router is now pointing to /exercises
 */
app.listen(port);
console.log('Server started and listening on port ' + port);








