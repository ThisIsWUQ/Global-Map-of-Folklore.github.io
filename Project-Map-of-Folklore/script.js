/* =================================================
           SENTENCE EMBEDDING IMPORT
        ================================================== */

        import {
            pipeline
        } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


        /* =================================================
           DATA
        ================================================== */

        let data = [];


        async function loadLegendData() {

            const response =
                await fetch("./legends.json");


            if (!response.ok) {

                throw new Error(
                    `Could not load legends.json (${response.status} ${response.statusText})`
                );

            }


            const json =
                await response.json();


            /*
             * Supports either:
             * 1. a direct JSON array: [ {...}, {...} ]
             * 2. an object wrapper: { "legends": [ {...}, {...} ] }
             */
            const stories =
                Array.isArray(json)
                    ? json
                    : json.legends;


            if (!Array.isArray(stories)) {

                throw new Error(
                    "legends.json must contain a JSON array, or an object with a 'legends' array."
                );

            }


            /*
             * Normalize fields so the rest of the existing app can
             * continue using id, name, text, location,
             * placeName, tags, etc.
             */
            data =
                stories.map(

                    (story, index) => ({

                        ...story,

                        id:
                            story.id ??
                            index + 1,

                        name:
                            story.name ??
                            "Untitled story",

                        text:
                            story.text ??
                            "",

                        location:
                            story.location ??
                            "",

                        placeName:
                            story.placeName ??
                            story.place_name ??
                            story.locationName ??
                            "",

                        tags:
                            Array.isArray(story.tags)
                                ? story.tags
                                : []

                    })

                );

        }


        /* =================================================
           GLOBAL STATE
        ================================================== */

        let map;

        let markerLayer;

        let selectedStory = null;

        let currentFilteredData = [];

        let selectedTags = new Set();

        let searchTerm = "";

        let semanticExtractor = null;

        let semanticEmbeddings = {};

        let embeddingsPromise = null;

        /*
         * One shared promise for corpus embedding calculation.
         * This prevents duplicate embedding jobs while still allowing
         * displaySemanticConnections() to await the result correctly.
         */
        let storyEmbeddingsPromise = null;

        let semanticModelReady = false;

        /*
         * Cache NLP results so selecting the same story again is instant.
         */
        const nlpCache = new Map();


        /*
         * PERFORMANCE: persist semantic vectors between page loads.
         * If legends.json has not changed, the browser can restore the
         * vectors instead of running the transformer model again.
         */
        const SEMANTIC_CACHE_VERSION = "folklore-semantic-v2";


        function semanticDatasetKey() {

            const signature =
                data.map(story =>

                    [
                        story.id,
                        story.text.length,
                        story.text.slice(0, 80)

                    ].join(":")

                ).join("|");


            let hash = 2166136261;


            for (let i = 0; i < signature.length; i++) {

                hash ^= signature.charCodeAt(i);

                hash = Math.imul(
                    hash,
                    16777619
                );

            }


            return (
                SEMANTIC_CACHE_VERSION +
                ":" +
                (hash >>> 0).toString(16)
            );

        }


        function restoreSemanticCache() {

            try {

                const raw =
                    localStorage.getItem(
                        semanticDatasetKey()
                    );


                if (!raw) {

                    return false;

                }


                const cached =
                    JSON.parse(raw);


                if (
                    !cached ||
                    Object.keys(cached).length !== data.length
                ) {

                    return false;

                }


                semanticEmbeddings =
                    cached;


                return true;

            }

            catch (error) {

                console.warn(
                    "Could not restore semantic cache:",
                    error
                );


                return false;

            }

        }


        function saveSemanticCache() {

            try {

                localStorage.setItem(

                    semanticDatasetKey(),

                    JSON.stringify(
                        semanticEmbeddings
                    )

                );

            }

            catch (error) {

                console.warn(
                    "Could not save semantic cache:",
                    error
                );

            }

        }


        /* =================================================
           STOPWORDS
        ================================================== */

        const stopwords = new Set([

            "the", "and", "for", "with", "that",
            "this", "from", "into", "about",
            "said", "were", "was", "are",
            "has", "have", "been", "their",
            "they", "them", "its", "who",
            "where", "which", "according",
            "such", "other", "than", "also",
            "over", "under", "after",
            "before", "while", "through",
            "these", "those", "very",
            "some", "more", "most",
            "family", "story", "stories",
            "legend", "legends",

            /* Personal pronouns */
            "i", "me", "my", "mine", "myself",
            "we", "us", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "itself",
            "they", "them", "their", "theirs", "themselves",

            /* Demonstrative pronouns */
            "this", "that", "these", "those",

            /* Relative/interrogative pronouns */
            "who", "whom", "whose", "which", "what",
            "whoever", "whomever", "whichever", "whatever",

            /* Indefinite pronouns */
            "anybody", "anyone", "anything",
            "everybody", "everyone", "everything",
            "nobody", "noone", "nothing",
            "somebody", "someone", "something",
            "another", "any", "both", "each", "either",
            "few", "many", "neither", "none", "one",
            "several", "all", "some"

        ]);


        /* =================================================
           HTML ESCAPE
        ================================================== */

        function escapeHTML(value) {

            return String(value)

                .replace(/&/g, "&amp;")

                .replace(/</g, "&lt;")

                .replace(/>/g, "&gt;")

                .replace(/"/g, "&quot;")

                .replace(/'/g, "&#039;");

        }


        /* =================================================
           PARSE COORDINATES

           Supports:
           40°40′18″N 73°24′54″W
        ================================================== */

        function parseDMS(coordinateString) {

            if (
                !coordinateString ||
                typeof coordinateString !== "string"
            ) {

                return null;

            }

            const regex =
                /(\d+)[°]\s*(\d+)?[′']?\s*(\d+)?[″"]?\s*([NS])\s*(\d+)[°]\s*(\d+)?[′']?\s*(\d+)?[″"]?\s*([EW])/i;

            const match =
                coordinateString.match(regex);

            if (!match) {

                return null;

            }

            let lat =
                Number(match[1]) +
                Number(match[2] || 0) / 60 +
                Number(match[3] || 0) / 3600;

            let lon =
                Number(match[5]) +
                Number(match[6] || 0) / 60 +
                Number(match[7] || 0) / 3600;

            if (
                match[4].toUpperCase() === "S"
            ) {

                lat = -lat;

            }

            if (
                match[8].toUpperCase() === "W"
            ) {

                lon = -lon;

            }

            return {

                lat: lat,

                lon: lon

            };

        }


        /* =================================================
           INITIALIZE MAP
        ================================================== */

        function initializeMap() {

            map = L.map("map", {

                worldCopyJump: true,

                scrollWheelZoom: true

            });


            const mapLayer =
                L.tileLayer(

                    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",

                    {

                        maxZoom: 19,

                        attribution:
                            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'

                    }

                );


            const satelliteLayer =
                L.tileLayer(

                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",

                    {

                        maxZoom: 19,

                        attribution:
                            '&copy; <a href="https://www.esri.com">ESRI</a>'

                    }

                );


            mapLayer.addTo(map);


            L.control.layers({

                "Map": mapLayer,

                "Satellite": satelliteLayer

            }).addTo(map);


            markerLayer =
                L.layerGroup().addTo(map);


            map.setView(

                [20, 0],

                2

            );


            updateMap(

                currentFilteredData

            );

        }


        /* =================================================
           UPDATE MAP
        ================================================== */

        function updateMap(stories) {

            markerLayer.clearLayers();


            const bounds = [];


            stories.forEach(

                story => {

                    const coordinates =
                        parseDMS(

                            story.location

                        );


                    if (!coordinates) {

                        return;

                    }


                    const marker =
                        L.circleMarker(

                            [

                                coordinates.lat,

                                coordinates.lon

                            ],

                            {

                                radius: 8,

                                fillColor:

                                    selectedStory &&
                                    selectedStory.id === story.id

                                        ? "#1769e0"

                                        : "#222222",

                                color: "#ffffff",

                                weight: 2,

                                fillOpacity: 0.95

                            }

                        );


                    marker.bindPopup(`

                        <strong>

                            ${escapeHTML(story.name)}

                        </strong>

                        <br><br>

                        ${escapeHTML(story.placeName || story.location)}

                        <br><br>

                        <button
                            id="select-marker-${story.id}"
                        >

                            Select story

                        </button>

                    `);


                    marker.on(

                        "click",

                        () => {

                            selectStory(

                                story

                            );

                        }

                    );


                    markerLayer.addLayer(

                        marker

                    );


                    bounds.push([

                        coordinates.lat,

                        coordinates.lon

                    ]);

                }

            );


            if (bounds.length > 0) {

                map.fitBounds(

                    bounds,

                    {

                        padding: [25, 25],

                        maxZoom: 4

                    }

                );

            }

        }


        /* =================================================
           TAG BUTTONS
        ================================================== */

        function createTagButtons() {

            const tagList =
                document.getElementById(

                    "tag-list"

                );


            const allTags = [

                ...new Set(

                    data.flatMap(

                        story => story.tags

                    )

                )

            ].sort(

                (a, b) => a.localeCompare(b)

            );


            tagList.innerHTML = "";


            allTags.forEach(

                tag => {

                    const button =
                        document.createElement(

                            "button"

                        );


                    button.className =
                        "tag-button";


                    button.textContent =
                        tag;


                    button.dataset.tag =
                        tag;


                    button.type =
                        "button";


                    button.addEventListener(

                        "click",

                        () => {

                            if (

                                selectedTags.has(tag)

                            ) {

                                selectedTags.delete(tag);

                                button.classList.remove(

                                    "active"

                                );

                            }

                            else {

                                selectedTags.add(tag);

                                button.classList.add(

                                    "active"

                                );

                            }


                            applyFilters();

                        }

                    );


                    tagList.appendChild(

                        button

                    );

                }

            );

        }


        /* =================================================
           FILTERING
        ================================================== */

        function applyFilters() {

            currentFilteredData =
                data.filter(

                    story => {

                        const matchesSearch =

                            story.name
                                .toLowerCase()
                                .includes(searchTerm) ||

                            story.text
                                .toLowerCase()
                                .includes(searchTerm) ||

                            story.tags.some(

                                tag =>

                                    tag
                                        .toLowerCase()
                                        .includes(searchTerm)

                            );


                        const matchesTags =

                            selectedTags.size === 0 ||

                            [...selectedTags].every(

                                tag =>

                                    story.tags.includes(tag)

                            );


                        return (

                            matchesSearch &&

                            matchesTags

                        );

                    }

                );


            renderStoryList(

                currentFilteredData

            );


            updateMap(

                currentFilteredData

            );

        }


        /* =================================================
           SEARCH
        ================================================== */

        document
            .getElementById("searchInput")
            .addEventListener(

                "input",

                event => {

                    searchTerm =
                        event.target.value
                            .toLowerCase()
                            .trim();


                    applyFilters();

                }

            );


        /* =================================================
           CLEAR FILTERS
        ================================================== */

        document
            .getElementById("clearFilters")
            .addEventListener(

                "click",

                () => {

                    selectedTags.clear();

                    searchTerm = "";


                    document
                        .getElementById("searchInput")
                        .value = "";


                    document
                        .querySelectorAll(".tag-button")
                        .forEach(

                            button =>

                                button.classList.remove(

                                    "active"

                                )

                        );


                    applyFilters();

                }

            );


        /* =================================================
           RENDER STORY LIST
        ================================================== */

        function renderStoryList(stories) {

            const container =
                document.getElementById(

                    "story-list"

                );


            const count =
                document.getElementById(

                    "story-count"

                );


            count.textContent =
                `${stories.length} ${stories.length === 1 ? "story" : "stories"}`;


            if (stories.length === 0) {

                container.innerHTML = `

                    <div class="no-stories">

                        No stories found.

                    </div>

                `;

                return;

            }


            container.innerHTML = "";


            stories.forEach(

                story => {

                    const item =
                        document.createElement(

                            "div"

                        );


                    item.className =
                        "story-item";


                    if (

                        selectedStory &&

                        selectedStory.id === story.id

                    ) {

                        item.classList.add(

                            "selected"

                        );

                    }


                    item.tabIndex = 0;

                    item.setAttribute(

                        "role",

                        "button"

                    );


                    item.setAttribute(

                        "aria-label",

                        `Select ${story.name}`

                    );


                    const tagsHTML =
                        story.tags

                            .map(

                                tag =>

                                    `<span class="story-tag">
                                        ${escapeHTML(tag)}
                                    </span>`

                            )

                            .join("");


                    item.innerHTML = `

                        <div class="story-name">

                            ${escapeHTML(story.name)}

                        </div>


                        <div class="story-tags">

                            ${tagsHTML}

                        </div>

                    `;


                    item.addEventListener(

                        "click",

                        () => {

                            selectStory(

                                story

                            );

                        }

                    );


                    item.addEventListener(

                        "keydown",

                        event => {

                            if (

                                event.key === "Enter" ||

                                event.key === " "

                            ) {

                                event.preventDefault();

                                selectStory(

                                    story

                                );

                            }

                        }

                    );


                    container.appendChild(

                        item

                    );

                }

            );

        }


        /* =================================================
           SELECT STORY
        ================================================== */

        function selectStory(story) {

            selectedStory =
                story;


            renderStoryList(

                currentFilteredData

            );


            displayDescription(

                story

            );


            displayNLP(

                story

            );


            updateMap(

                currentFilteredData

            );

        }


        /* =================================================
           DESCRIPTION
        ================================================== */

        function displayDescription(story) {

            const content =
                document.getElementById(

                    "descriptionContent"

                );


            content.classList.remove(

                "placeholder"

            );


            content.innerHTML = `

                <div class="description-title">

                    ${escapeHTML(story.name)}

                </div>


                <div class="description-text">

                    ${escapeHTML(story.text).replace(/\n/g, '<br>')}

                </div>


                <div class="location">

                    📍 ${escapeHTML(story.placeName || story.location)}

                </div>

            `;

        }


        /* =================================================
           TOP 5 KEYWORDS
        ================================================== */

        function getTopKeywords(text) {

            const tokens =
                text

                    .toLowerCase()

                    .match(/\b[a-z]+\b/g) || [];


            const frequency = {};


            tokens.forEach(

                word => {

                    if (

                        stopwords.has(word) ||

                        word.length < 3

                    ) {

                        return;

                    }


                    frequency[word] =

                        (frequency[word] || 0) + 1;

                }

            );


            return Object

                .entries(frequency)

                .sort(

                    (a, b) => {

                        if (b[1] !== a[1]) {

                            return b[1] - a[1];

                        }


                        return a[0].localeCompare(b[0]);

                    }

                )

                .slice(0, 5);

        }


        /* =================================================
           SIMPLE NER

           This is a lightweight rule-based NER.
        ================================================== */

        function getEntities(text) {

            const entities = [];


            const knownPlaces = [

                "Amityville",

                "New York",

                "Tennessee",

                "Adams",

                "Hamelin",

                "America",

                "United States"

            ];


            knownPlaces.forEach(

                place => {

                    if (

                        text
                            .toLowerCase()
                            .includes(

                                place.toLowerCase()

                            )

                    ) {

                        entities.push({

                            type: "Place",

                            text: place

                        });

                    }

                }

            );


            return entities.filter(

                (entity, index, array) =>

                    index ===

                    array.findIndex(

                        other =>

                            other.type === entity.type &&

                            other.text === entity.text

                    )

            );

        }


        /* =================================================
           DISPLAY NLP
        ================================================== */

        function displayNLP(story) {

            const nlpContent =
                document.getElementById(

                    "nlpContent"

                );


            const text = story.text;


            let cachedNLP =
                nlpCache.get(story.id);


            if (!cachedNLP) {

                cachedNLP = {

                    topKeywords:
                        getTopKeywords(text),

                    entities:
                        getEntities(text)

                };


                nlpCache.set(
                    story.id,
                    cachedNLP
                );

            }


            const topKeywords =
                cachedNLP.topKeywords;


            const entities =
                cachedNLP.entities;


            let html = "";


            /* TOP KEYWORDS */

            html += `

                <div class="nlp-section">

                    <div class="nlp-heading">

                        Top 5 keywords

                    </div>

            `;


            if (topKeywords.length === 0) {

                html += `

                    <div class="placeholder">

                        No keywords found.

                    </div>

                `;

            }

            else {

                topKeywords.forEach(

                    ([word, frequency]) => {

                        html += `

                            <div class="keyword-row">

                                <span>

                                    ${escapeHTML(word)}

                                </span>

                                <span class="keyword-frequency">

                                    ${frequency}

                                </span>

                            </div>

                        `;

                    }

                );

            }


            html += `

                </div>

            `;


            /* NAMED ENTITIES */

            html += `

                <div class="nlp-section">

                    <div class="nlp-heading">

                        Named entities

                    </div>

            `;


            if (entities.length === 0) {

                html += `

                    <div class="placeholder">

                        No named entities detected.

                    </div>

                `;

            }

            else {

                entities.forEach(

                    entity => {

                        html += `

                            <div class="entity-row">

                                <span class="entity-type">

                                    ${escapeHTML(entity.type)}

                                </span>

                                <span>

                                    ${escapeHTML(entity.text)}

                                </span>

                            </div>

                        `;

                    }

                );

            }


            html += `

                </div>

            `;


            nlpContent.innerHTML =
                html;


            nlpContent.classList.remove(

                "placeholder"

            );


            displaySemanticConnections(

                story

            );

        }


        /* =================================================
           LOAD SENTENCE EMBEDDING MODEL
        ================================================== */

        async function loadSemanticModel() {

            if (semanticModelReady) {

                return semanticExtractor;

            }


            if (embeddingsPromise) {

                return embeddingsPromise;

            }


            const status =
                document.getElementById(

                    "semantic-status"

                );


            status.textContent =
                "Preparing semantic analysis...";


            /*
             * PERFORMANCE:
             * Prefer WebGPU on supported browsers. This lets the
             * sentence-embedding model use the computer's GPU.
             * If WebGPU/model loading fails, automatically fall
             * back to the compatible quantized WASM version.
             */
            const createSemanticPipeline =
                async () => {

                    if (navigator.gpu) {

                        try {

                            console.log(
                                "Trying WebGPU semantic acceleration..."
                            );


                            return await pipeline(

                                "feature-extraction",

                                "Xenova/all-MiniLM-L6-v2",

                                {
                                    device: "webgpu",
                                    dtype: "fp16"
                                }

                            );

                        }

                        catch (webgpuError) {

                            console.warn(
                                "WebGPU unavailable for this model; using WASM.",
                                webgpuError
                            );

                        }

                    }


                    return pipeline(

                        "feature-extraction",

                        "Xenova/all-MiniLM-L6-v2",

                        {
                            device: "wasm",
                            dtype: "q8"
                        }

                    );

                };


            embeddingsPromise =
                createSemanticPipeline();


            try {

                semanticExtractor =
                    await embeddingsPromise;


                semanticModelReady =
                    true;


                return semanticExtractor;

            }

            catch (error) {

                console.error(

                    "Could not load semantic model:",

                    error

                );


                embeddingsPromise = null;


                throw error;

            }

        }


        /* =================================================
           CALCULATE EMBEDDINGS
        ================================================== */

        async function calculateStoryEmbeddings() {

            /*
             * Fastest path: restore vectors saved by a previous visit.
             * This avoids loading/running the AI model when the JSON
             * dataset is unchanged.
             */
            if (

                Object.keys(
                    semanticEmbeddings
                ).length === 0

            ) {

                restoreSemanticCache();

            }


            /*
             * Fast path: embeddings have already been calculated.
             * The previous version re-embedded EVERY legend each
             * time a story was selected.
             */
            if (

                Object.keys(
                    semanticEmbeddings
                ).length === data.length

            ) {

                return semanticEmbeddings;

            }


            /*
             * If calculation is already running, reuse the same
             * promise instead of starting another expensive job.
             */
            if (storyEmbeddingsPromise) {

                return storyEmbeddingsPromise;

            }


            storyEmbeddingsPromise =
                (async () => {

                    const extractor =
                        await loadSemanticModel();


                    const texts =
                        data.map(

                            story =>
                                story.text

                        );


                    const output =
                        await extractor(

                            texts,

                            {

                                pooling: "mean",

                                normalize: true

                            }

                        );


                    const vectors =
                        output.tolist();


                    data.forEach(

                        (story, index) => {

                            semanticEmbeddings[story.id] =
                                vectors[index];

                        }

                    );


                    /*
                     * Persist the vectors. On later page loads the
                     * semantic graph can be drawn without recomputing
                     * embeddings, as long as legends.json is unchanged.
                     */
                    saveSemanticCache();


                    return semanticEmbeddings;

                })();


            try {

                return await storyEmbeddingsPromise;

            }

            catch (error) {

                /*
                 * Allow a later retry if model inference failed.
                 */
                storyEmbeddingsPromise = null;

                throw error;

            }

        }


        /* =================================================
           COSINE SIMILARITY
        ================================================== */

        function cosineSimilarity(a, b) {

            let dot = 0;

            let magnitudeA = 0;

            let magnitudeB = 0;


            for (

                let i = 0;

                i < a.length;

                i++

            ) {

                dot += a[i] * b[i];

                magnitudeA += a[i] * a[i];

                magnitudeB += b[i] * b[i];

            }


            magnitudeA =
                Math.sqrt(magnitudeA);


            magnitudeB =
                Math.sqrt(magnitudeB);


            if (

                magnitudeA === 0 ||

                magnitudeB === 0

            ) {

                return 0;

            }


            return (

                dot /

                (

                    magnitudeA *

                    magnitudeB

                )

            );

        }


        /* =================================================
           SEMANTIC CONNECTIONS
        ================================================== */

        async function displaySemanticConnections(story) {

            const status =
                document.getElementById(

                    "semantic-status"

                );


            const svg =
                document.getElementById(

                    "semantic-map"

                );


            status.className =
                "semantic-status";


            status.textContent =
                "Calculating semantic connections...";


            svg.innerHTML = "";


            try {

                await calculateStoryEmbeddings();


                const selectedVector =
                    semanticEmbeddings[story.id];


                if (!selectedVector) {

                    throw new Error(

                        "Selected story has no embedding."

                    );

                }


                const similarities =

                    data

                        .filter(

                            item =>

                                item.id !== story.id

                        )

                        .map(

                            item => {

                                const similarity =

                                    cosineSimilarity(

                                        selectedVector,

                                        semanticEmbeddings[item.id]

                                    );


                                return {

                                    story: item,

                                    similarity: similarity

                                };

                            }

                        )

                        .sort(

                            (a, b) =>

                                b.similarity -

                                a.similarity

                        );


                drawSemanticMap(

                    story,

                    similarities

                );


                status.textContent =
                    "Higher similarity means a closer point.";

            }

            catch (error) {

                console.error(

                    "Semantic connections failed:",

                    error

                );


                status.className =
                    "semantic-status semantic-error";


                status.textContent =
                    "Could not calculate semantic connections. Check the browser console.";

            }

        }


        /* =================================================
           DRAW SEMANTIC MAP
        ================================================== */

        function drawSemanticMap(selectedStory, similarities) {

            const svg =
                document.getElementById(

                    "semantic-map"

                );


            svg.innerHTML = "";


            const width = 460;

            const height = 300;


            const centerX =
                width / 2;

            const centerY =
                height / 2;


            const connections =
                similarities.slice(0, 5);


            if (connections.length === 0) {

                return;

            }


            const maxSimilarity =
                Math.max(

                    ...connections.map(

                        item => item.similarity

                    )

                );


            const minSimilarity =
                Math.min(

                    ...connections.map(

                        item => item.similarity

                    )

                );


            function createSVGElement(tag, attributes) {

                const element =
                    document.createElementNS(

                        "http://www.w3.org/2000/svg",

                        tag

                    );


                Object.entries(attributes).forEach(

                    ([key, value]) => {

                        element.setAttribute(

                            key,

                            value

                        );

                    }

                );


                return element;

            }


            /* =================================================
               OTHER STORIES
            ================================================== */

            connections.forEach(

                (item, index) => {

                    const angle =

                        (

                            index /

                            connections.length

                        ) *

                        Math.PI *

                        2;


                    let normalized;


                    if (

                        maxSimilarity === minSimilarity

                    ) {

                        normalized = 0.5;

                    }

                    else {

                        normalized =

                            (

                                maxSimilarity -

                                item.similarity

                            ) /

                            (

                                maxSimilarity -

                                minSimilarity

                            );

                    }


                    const minRadius = 88;

                    const maxRadius = 128;


                    const radius =

                        minRadius +

                        normalized *

                        (

                            maxRadius -

                            minRadius

                        );


                    const x =

                        centerX +

                        Math.cos(angle) *

                        radius;


                    const y =

                        centerY +

                        Math.sin(angle) *

                        radius;


                    /* CONNECTION LINE */

                    const line =
                        createSVGElement(

                            "line",

                            {

                                x1: centerX,

                                y1: centerY,

                                x2: x,

                                y2: y,

                                class: "semantic-line"

                            }

                        );


                    svg.appendChild(line);


                    /* SIMILARITY VALUE */

                    const similarityText =
                        createSVGElement(

                            "text",

                            {

                                x:

                                    centerX +

                                    (x - centerX) *

                                    0.68,

                                y:

                                    centerY +

                                    (y - centerY) *

                                    0.68 - 5,

                                class:

                                    "semantic-similarity",

                                "text-anchor":

                                    "middle"

                            }

                        );


                    similarityText.textContent =

                        item.similarity.toFixed(2);


                    svg.appendChild(

                        similarityText

                    );


                    /* STORY NODE */

                    const group =
                        createSVGElement(

                            "g",

                            {

                                class: "semantic-node"

                            }

                        );


                    const circle =
                        createSVGElement(

                            "circle",

                            {

                                cx: x,

                                cy: y,

                                r: 7

                            }

                        );


                    group.appendChild(

                        circle

                    );


                    /* LABEL */

                    const label =
                        createSVGElement(

                            "text",

                            {

                                x: x,

                                y: y + (index % 2 === 0 ? -18 : 25),

                                class:

                                    "semantic-label",

                                "text-anchor":

                                    "middle"

                            }

                        );


                    let labelText =
                        item.story.name;


                    if (labelText.length > 26) {

                        labelText =

                            labelText.substring(

                                0,

                                23

                            ) + "...";

                    }


                    label.textContent =
                        labelText;


                    group.appendChild(

                        label

                    );


                    /* TOOLTIP */

                    const title =
                        createSVGElement(

                            "title",

                            {}

                        );


                    title.textContent =

                        `${item.story.name}
Similarity: ${item.similarity.toFixed(3)}`;


                    group.appendChild(

                        title

                    );


                    /* CLICK TO SELECT */

                    group.addEventListener(

                        "click",

                        () => {

                            selectStory(

                                item.story

                            );

                        }

                    );


                    svg.appendChild(

                        group

                    );

                }

            );


            /* =================================================
               SELECTED STORY AT CENTER
            ================================================== */

            const selectedGroup =
                createSVGElement(

                    "g",

                    {

                        class:

                            "semantic-node selected"

                    }

                );


            const selectedCircle =
                createSVGElement(

                    "circle",

                    {

                        cx: centerX,

                        cy: centerY,

                        r: 10

                    }

                );


            selectedGroup.appendChild(

                selectedCircle

            );


            const selectedLabel =
                createSVGElement(

                    "text",

                    {

                        x: centerX,

                        y: centerY + 30,

                        class:

                            "semantic-label",

                        "text-anchor":

                            "middle"

                    }

                );


            selectedLabel.textContent =

                selectedStory.name;


            selectedGroup.appendChild(

                selectedLabel

            );


            const selectedTitle =
                createSVGElement(

                    "title",

                    {}

                );


            selectedTitle.textContent =

                selectedStory.name;


            selectedGroup.appendChild(

                selectedTitle

            );


            svg.appendChild(

                selectedGroup

            );

        }


        /* =================================================
           INITIALIZE
        ================================================== */

        async function initializeApp() {

            try {

                await loadLegendData();


                currentFilteredData =
                    [...data];


                initializeMap();


                createTagButtons();


                renderStoryList(
                    data
                );


                /*
                 * PERFORMANCE:
                 * Render the interface first. Then warm the semantic
                 * model and corpus embeddings in browser idle time.
                 * A story click can reuse this same background job.
                 */
                const hasCachedEmbeddings =
                    restoreSemanticCache();


                const warmSemanticEngine =
                    () => {

                        /*
                         * If vectors were restored from localStorage,
                         * there is nothing expensive to prepare.
                         */
                        if (hasCachedEmbeddings) {

                            const status =
                                document.getElementById(
                                    "semantic-status"
                                );


                            if (
                                status &&
                                !selectedStory
                            ) {

                                status.textContent =
                                    "Select a story to view semantic connections.";

                            }


                            console.log(
                                "Semantic embeddings restored from browser cache."
                            );


                            return;

                        }


                        calculateStoryEmbeddings()

                            .then(

                                () => {

                                    const status =
                                        document.getElementById(
                                            "semantic-status"
                                        );


                                    if (
                                        status &&
                                        !selectedStory
                                    ) {

                                        status.textContent =
                                            "Select a story to view semantic connections.";

                                    }


                                    console.log(
                                        "Semantic embeddings ready."
                                    );

                                }

                            )

                            .catch(

                                error => {

                                    console.warn(
                                        "Background semantic preparation failed:",
                                        error
                                    );

                                }

                            );

                    };


                if ("requestIdleCallback" in window) {

                    requestIdleCallback(

                        warmSemanticEngine,

                        {
                            timeout: 2500
                        }

                    );

                }

                else {

                    setTimeout(
                        warmSemanticEngine,
                        1200
                    );

                }


                /*
                 * Ensure Leaflet correctly measures its
                 * grid container.
                 */
                setTimeout(

                    () => {

                        if (map) {

                            map.invalidateSize();

                        }

                    },

                    300

                );

            } catch (error) {

                console.error(
                    "Could not initialize Folklore Map:",
                    error
                );


                const storyList =
                    document.getElementById(
                        "story-list"
                    );


                if (storyList) {

                    storyList.innerHTML = `
                        <div style="padding:16px; color:#a33; line-height:1.5;">
                            Could not load <strong>legends.json</strong>.<br><br>
                            Make sure legends.json is in the same folder as this HTML file
                            and open the project through a local web server rather than
                            directly with <code>file://</code>.
                        </div>
                    `;

                }

            }

        }


        initializeApp();
